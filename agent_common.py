"""Shared pieces for the agent demos: the Claude client, one model call, token
cost, and the agent loop used by demos 2 and 3.

Demo 1 writes the same loop out in full so it can be read in one place;
run_agent() below is that loop with hooks for tracing and stopping.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# One line to change the model. Effort trades thoroughness for tokens: "low"
# keeps the demos fast and cheap; try "medium" or "high" and compare the runs.
MODEL = "claude-opus-5-5"
EFFORT = "low"

# US dollars per million tokens for MODEL (cache writes use the 5-minute TTL).
# Check https://platform.claude.com/docs/en/about-claude/pricing before relying on them.
PRICES = {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20}


def make_client():
    """An Anthropic client using ANTHROPIC_API_KEY from the environment or from .env."""
    import anthropic
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        pass
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and not key.startswith("sk-ant-"):
        raise SystemExit(f"ANTHROPIC_API_KEY starts with '{key[:7]}', but Anthropic keys start with 'sk-ant-'.")
    if not key and not (os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_PROFILE")
                        or os.path.isdir(os.path.expanduser("~/.config/anthropic"))):
        raise SystemExit(f"No Anthropic API key found. Put ANTHROPIC_API_KEY=sk-ant-... in {HERE / '.env'}.")
    return anthropic.Anthropic()  # falls back to the SDK's own lookup (ant auth login) when no key is set


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    @classmethod
    def of(cls, u):
        return cls(u.input_tokens, u.output_tokens, u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0)

    def __iadd__(self, other):
        self.input += other.input
        self.output += other.output
        self.cache_write += other.cache_write
        self.cache_read += other.cache_read
        return self

    @property
    def cost(self):
        return sum(getattr(self, k) * PRICES[k] for k in PRICES) / 1e6

    def as_dict(self):
        return {"input": self.input, "output": self.output, "cache_write": self.cache_write,
                "cache_read": self.cache_read, "cost_usd": round(self.cost, 5)}


def call_claude(client, *, system, messages, tools=None, output_format=None, max_tokens=16000):
    """One request to the Messages API. Returns (response, seconds).

    - Adaptive thinking is always on for this model; display="summarized" returns a
      readable summary of the reasoning before each step, which the demos print.
    - cache_control caches the request prefix (system prompt, tools, earlier turns),
      so each loop step pays full price only for what is new.
    - fallbacks="default": if a safety classifier declines, the API retries the same
      request on another model inside the same call.
    """
    output_config = {"effort": EFFORT}
    if output_format:
        output_config["format"] = output_format
    params = dict(model=MODEL, max_tokens=max_tokens, system=system, messages=messages,
                  thinking={"type": "adaptive", "display": "summarized"}, output_config=output_config,
                  cache_control={"type": "ephemeral"},
                  betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if tools:
        params["tools"] = tools
    start = time.perf_counter()
    response = client.beta.messages.create(**params)
    return response, time.perf_counter() - start


def text_of(response):
    return "".join(b.text for b in response.content if b.type == "text")


def thinking_of(response):
    return " ".join(b.thinking for b in response.content if b.type == "thinking" and b.thinking).strip()


class AgentStopped(Exception):
    """Raised by an on_event hook to stop the agent, for example when a budget runs out."""


@dataclass
class AgentResult:
    answer: str
    stop: str                       # end_turn, max_steps, refusal, max_tokens, or a hook's reason
    steps: int
    tool_calls: int
    tool_errors: int
    usage: Usage
    seconds: float
    messages: list = field(repr=False)


def run_agent(client, *, system, task, tools, execute_tool, max_steps=12, output_format=None, on_event=None):
    """The agent loop: call the model, run the tools it asks for, send back the
    results, and repeat until it answers without asking for a tool.

    execute_tool(name, args) returns a string or raises; a raised error goes back
    to the model as a tool_result with is_error set, so the model can recover.
    on_event(kind, **data) is called after every model call ("llm") and tool call
    ("tool"); it may raise AgentStopped to end the run.
    """
    emit = on_event or (lambda kind, **data: None)
    messages = [{"role": "user", "content": task}]
    usage, tool_calls, tool_errors = Usage(), 0, 0
    started = time.perf_counter()

    def result(answer, stop, step):
        return AgentResult(answer, stop, step, tool_calls, tool_errors, usage,
                           time.perf_counter() - started, messages)

    for step in range(1, max_steps + 1):
        try:
            response, seconds = call_claude(client, system=system, messages=messages, tools=tools,
                                            output_format=output_format)
            step_usage = Usage.of(response.usage)
            usage += step_usage
            messages.append({"role": "assistant", "content": response.content})
            emit("llm", step=step, response=response, seconds=seconds, usage=step_usage, total=usage)

            if response.stop_reason in ("refusal", "max_tokens"):
                return result(text_of(response), response.stop_reason, step)
            if response.stop_reason != "tool_use":
                return result(text_of(response), "end_turn", step)

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                start = time.perf_counter()
                try:
                    output, is_error = execute_tool(block.name, block.input), False
                except AgentStopped:
                    raise
                except Exception as exc:  # any tool failure goes back to the model, not up the stack
                    output, is_error = f"{type(exc).__name__}: {exc}", True
                    tool_errors += 1
                emit("tool", step=step, name=block.name, args=block.input, output=output,
                     is_error=is_error, seconds=time.perf_counter() - start)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output,
                                **({"is_error": True} if is_error else {})})
            # All results from one turn go back in a single user message.
            messages.append({"role": "user", "content": results})
        except AgentStopped as stop:
            return result("", str(stop), step)
    return result("", "max_steps", max_steps)


def jsonable(messages):
    """The conversation as plain JSON, for saving a transcript.

    Thinking-block signatures are dropped: they are only needed to send the
    conversation back to the API, and they make the file hard to read.
    """
    out = []
    for m in messages:
        content = m["content"]
        if not isinstance(content, str):
            content = [b if isinstance(b, dict) else b.model_dump(mode="json", exclude_none=True) for b in content]
            for b in content:
                b.pop("signature", None)
        out.append({"role": m["role"], "content": content})
    return out


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def short(text, limit=220):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."
