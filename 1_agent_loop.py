"""Demo 1: the agent loop, written out.

An agent is a model that chooses its own next step. In code that is a loop:

  1. Send the task, the system prompt and the tool definitions to the model.
  2. If the reply asks for tools (stop_reason "tool_use"), run them in your own
     code and send the results back.
  3. Repeat until the reply asks for no tool (stop_reason "end_turn"), or a step
     limit is reached.

The model never runs anything itself: it only asks. Your code decides what a
tool does, what it may touch, and when the loop stops. Agent frameworks, and the
SDK's tool runner, are this loop with conveniences added.

The task is an incident in the simulated system in ops_system.py; the tools are
the read-only checks a System Monitoring notebook builds (health, metrics, drift
tests, canaries, deploys, logs).

    python 1_agent_loop.py                          # the upstream_outage scenario
    python 1_agent_loop.py --scenario drift_units   # or healthy, bad_deploy, noise

The full conversation, including every tool call and result, is saved to
runs/1_agent_loop_<scenario>.json.
"""

import argparse
import json

from agent_common import EFFORT, MODEL, RUNS, Usage, call_claude, jsonable, make_client, save_json, short, \
    text_of, thinking_of
from ops_system import READ_TOOLS, SCENARIOS, OpsSystem, ToolError, system_prompt

MAX_STEPS = 12   # a loop must always have a way to end


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="upstream_outage", choices=SCENARIOS)
    args = parser.parse_args()

    ops = OpsSystem(args.scenario)
    client = make_client()
    task = f"{ops.trigger()}\n\nFind the root cause and say what should be done about it."
    messages = [{"role": "user", "content": task}]
    total = Usage()

    print(f"Model {MODEL}, effort {EFFORT}, scenario {args.scenario}")
    print(f"Task: {task}")

    for step in range(1, MAX_STEPS + 1):
        # 1. Ask the model what to do next.
        response, seconds = call_claude(client, system=system_prompt(), messages=messages, tools=READ_TOOLS)
        usage = Usage.of(response.usage)
        total += usage
        # Keep the whole reply, thinking blocks included: the next request must
        # carry them back unchanged.
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n[step {step}] {seconds:.1f} s, {usage.input + usage.cache_read + usage.cache_write} tokens in "
              f"({usage.cache_read} from cache), {usage.output} out, stop_reason={response.stop_reason}")
        if thinking := thinking_of(response):
            print(f"  reasoning: {short(thinking, 320)}")

        # 3. No tool requested: the model has answered.
        if response.stop_reason != "tool_use":
            break

        # 2. Run every tool the model asked for, and send all the results back
        #    in one message. A failed tool is reported, not raised, so the
        #    model can try something else.
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output, is_error = ops.execute(block.name, block.input), False
            except ToolError as exc:
                output, is_error = str(exc), True
            print(f"  tool  {block.name}({json.dumps(block.input)})")
            print(f"     -> {'ERROR ' if is_error else ''}{short(output)}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output,
                            **({"is_error": True} if is_error else {})})
        messages.append({"role": "user", "content": results})
    else:
        print(f"\nStopped after {MAX_STEPS} steps without an answer.")

    if response.stop_reason == "refusal":
        print("\nThe model declined this request.")
    print("\n=== Answer ===")
    print(text_of(response))
    print(f"\n{step} model calls, {total.input + total.cache_read + total.cache_write} input tokens "
          f"({total.cache_read} from cache), {total.output} output tokens, about ${total.cost:.3f}")
    print(f"Ground truth: {SCENARIOS[args.scenario]['about']}")

    path = save_json(RUNS / f"1_agent_loop_{args.scenario}.json",
                     {"model": MODEL, "effort": EFFORT, "scenario": args.scenario, "usage": total.as_dict(),
                      "messages": jsonable(messages)})
    print(f"Transcript: {path.relative_to(RUNS.parent)}")


if __name__ == "__main__":
    main()
