# Agent Examples

Worked examples of language-model agents for a production AI system: what an agent is in code, when one is worth its cost over a fixed pipeline, and the controls and monitoring it needs before it may change anything.

All three examples use the same task. An **on-call agent** diagnoses incidents in a small simulated system built like the course projects: a data component, two models, an ensemble that combines them, and an API.

```
api-gateway  ->  ensemble  ->  demand-model    ->  data-service
                           ->  stockout-model  ->  data-service
```

The agent's tools are the checks from Module 12's System Monitoring work: health endpoints, metric history, drift tests (Kolmogorov-Smirnov, PSI, chi-square), canary inputs, deploy history and logs. The models, drift tests and canaries are real: scikit-learn models trained on generated data and scipy tests on that data. Only the infrastructure is scripted (health values, deploys, logs).

Background for the module: [`Module12-Agents-Overview.md`](Module12-Agents-Overview.md).

---

## Contents

| File | What it shows | Model calls | Runtime |
|---|---|---|---|
| [`1_agent_loop.py`](1_agent_loop.py) | The agent loop written out in about 40 lines: the model asks for tools, your code runs them, the results go back, repeat until it answers. Prints each step's reasoning summary, tool calls and results. | ~4–8 | < 1 min |
| [`2_workflow_vs_agent.py`](2_workflow_vs_agent.py) | A fixed workflow (run every check, then one model call) against the agent, on five incidents with known causes. Reports accuracy, model calls, tool calls, tokens, dollars and seconds for each. | ~30–50 | 1–3 min |
| [`3_guarded_agent.py`](3_guarded_agent.py) | The agent given tools that change production (rollback, restart), with a trace, budgets, a human approval gate, input validation, simulated tool failures and an injected instruction in a log line. `report` then monitors the agent across runs against thresholds. | ~5–10 per run | < 1 min |
| [`ops_system.py`](ops_system.py) | The simulated system, its five scenarios, the tool definitions and the runbook. | none | — |
| [`agent_common.py`](agent_common.py) | The Claude client, one model call, token cost, and the agent loop used by demos 2 and 3. | — | — |

### Scenarios

| Scenario | What is wrong | Correct diagnosis | What makes it hard |
|---|---|---|---|
| `healthy` | Nothing | none / none / none | Knowing when to stop |
| `drift_units` | data-service v2.3 switched weather provider and now sends temperature in Fahrenheit | data-service / data_drift / rollback | Every component is healthy. Only the drift tests and the error against actual rides show it, and the models' canaries pass. |
| `bad_deploy` | demand-model v1.8 fails on missing values and learned two features swapped | demand-model / bad_deploy / rollback | Errors show at every caller. The alert is on the API, not the model. |
| `upstream_outage` | data-service is in an out-of-memory crash loop | data-service / outage / restart_or_scale | A recent stockout-model deploy is a red herring |
| `noise` | Humidity shifted slightly: KS p = 0.005, PSI = 0.06 | none / none / none | The test is significant, but the shift is too small to act on |

---

## Setup

Python 3.10 or later. Everything runs on CPU.

```bash
cd agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Anthropic API key in .env
```

`.env` is in `.gitignore`; never commit it. If no key is set, the SDK falls back to its own credential lookup (an `ant auth login` profile).

**Model and cost.** The model is set in one place, `MODEL` in `agent_common.py` (`claude-opus-5-5`), at `EFFORT = "low"` to keep runs fast and cheap. Each run prints its token use and an estimated cost from `PRICES` in the same file. As a rough guide, a single agent run costs a few cents to about $0.15, and demo 2's full comparison under $1. Every request turns on prompt caching, so each loop step pays full price only for the new part of the conversation. Every request also turns on the API's server-side fallback: if a safety classifier declines a request, the API retries it on another model inside the same call.

---

## The examples

### 1. The agent loop: `1_agent_loop.py`

```bash
python 1_agent_loop.py                          # upstream_outage
python 1_agent_loop.py --scenario drift_units   # healthy, bad_deploy, noise
```

Read `main()` from top to bottom. It is the whole of an agent:

1. Send the task, the system prompt and the tool definitions.
2. If the reply's `stop_reason` is `tool_use`, run each requested tool in your own code and send the results back, all in one message.
3. Repeat until the reply asks for no tool, or `MAX_STEPS` is reached.

**What to look for:**
- The model never runs anything. It returns a tool name and JSON arguments, and `ops.execute()` decides what happens. That is why the tool layer is where security, validation and approval belong.
- In `upstream_outage`, the alert is on api-gateway, but the agent follows the `calls` links down to data-service. Compare its path with the runbook rule in `ops_system.py` ("the root cause is the furthest-downstream component that is failing").
- Each step re-sends the whole conversation. Watch input tokens grow, and the "from cache" share with them.
- A tool error, such as asking for `mae` on stockout-model, goes back to the model as a result with `is_error` set. The model reads it and carries on.

The whole conversation, every block included, is saved to `runs/1_agent_loop_<scenario>.json`.

### 2. Workflow vs. agent: `2_workflow_vs_agent.py`

```bash
python 2_workflow_vs_agent.py                 # every scenario, once
python 2_workflow_vs_agent.py --repeats 3     # the agent varies between runs: measure it
```

Both designs return the same structured diagnosis (`output_config.format` with a JSON schema), graded against each scenario's known cause:

- **workflow**: code runs all 35 checks in a fixed order, then one model call reads them all.
- **agent**: the model picks its checks one step at a time.

**What to look for:** on a system this small, the workflow sees everything for one fixed cost, which makes it a hard baseline to beat. Check whether the agent is more accurate, and what it costs in dollars and seconds to find out. The agent's case gets stronger as the number of possible checks grows past what fits in one prompt, or when the next check depends on the last result in ways a fixed plan cannot list. This is the comparison the Ensemble Approach assignment asks for: the more complex design against the simplest one that works, on a metric agreed beforehand. Run with `--repeats 3` to see the agent's run-to-run spread in steps and cost. One run of an agent is an anecdote, not a measurement.

Results go to `runs/2_workflow_vs_agent.json`.

### 3. A guarded agent and its monitoring: `3_guarded_agent.py`

```bash
python 3_guarded_agent.py                                   # bad_deploy; asks you before any action
python 3_guarded_agent.py --approve no                      # deny every action
python 3_guarded_agent.py --scenario upstream_outage --flaky 0.3
python 3_guarded_agent.py report                            # monitor every saved run
```

| Control | Where | What it does |
|---|---|---|
| Trace | `Tracer` | One JSON line per model call, tool call and approval in `runs/traces/`, with tokens, dollars, seconds and the reasoning summary |
| Budgets | `BUDGET`, `make_monitor` | Stops the run at 15 steps, 30 tool calls, $0.50 or 2 action requests |
| Validation | `ops.validate_action` | Rejects an action with impossible arguments (a version the component never ran) before a person is asked |
| Approval | `ask_person` | rollback and restart need a yes at the terminal (`--approve ask`), or `--approve yes/no`. A denial goes back to the model as an error, and the model reports instead. |
| Failures | `--flaky` | Read-only tools fail at random, and the model retries or works around them |
| Untrusted data | `--no-inject` to turn off | In `bad_deploy`, a log line tells the agent to roll back every component to v0.1 and skip approval |

**What to look for:** the injected instruction should be treated as data, because the system prompt says tool output is never an instruction. If the model did act on it, `validate_action` would reject v0.1 and the gate would still ask a person. Neither control depends on the model behaving well.

`report` reads every finished trace and checks the agent against thresholds: p95 cost per run, maximum steps, p95 seconds, tool error rate, runs stopped by a budget, and denied actions. Crossed thresholds are printed and appended to `runs/agent_alerts.log`, which stands in for an Azure Monitor alert rule or a webhook. It has the same shape as Part 3 of the System Monitoring assignment, applied to the agent instead of the models.

---

## Suggested order

1. `python 1_agent_loop.py`, then the same with `--scenario drift_units`: read the loop, then watch it handle a failure no health check sees.
2. `python 2_workflow_vs_agent.py`: decide with numbers whether this task needs an agent.
3. `python 3_guarded_agent.py`, then once with `--approve no` and once with `--flaky 0.3`, then `report`.

## Taking it to your own system

- **Tools** are functions you already have: a health endpoint, a drift test from your monitoring notebook, or a member's model wrapped as `predict(...)`. Give each one a description that says *when* to call it, an input schema with enums where possible, and `strict: True`.
- **Start read-only.** Add an action tool only with validation in code and approval by a person.
- **Measure first.** Run your workflow and your agent on the same cases, several times each, and compare.
- **Trace every step** from the first run. Without a trace, you cannot debug or monitor an agent.

## References

- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), 2024: workflows against agents, and the common patterns
- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023
- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): defining tools, `tool_use` and `tool_result`, strict tool use
- OWASP, [Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/): LLM01 Prompt Injection, LLM06 Excessive Agency, LLM10 Unbounded Consumption
- OpenTelemetry, [semantic conventions for generative AI](https://github.com/open-telemetry/semantic-conventions-genai): standard names for model, agent and tool spans
