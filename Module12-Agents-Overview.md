# Agents: When the Model Chooses the Next Step

An ensemble combines models in a way you fix in code: average these outputs, stack those, route by this rule. An **agent** hands that choice to a language model. The model is given a task and a set of tools, functions your code runs, which can include your team's models and your monitoring checks. It decides which tool to call next from what the last one returned, and stops when it can answer. That makes an agent the most flexible way to integrate models and checks, and also the hardest to predict, test and pay for. This section covers what an agent is in code, when it is worth its cost over a fixed pipeline, and what it needs before it may change a production system.

## Objective

12.5 Decide whether a fixed workflow or an agent should integrate a set of models or checks, and measure the choice. Name the controls an agent needs in production: step and cost budgets, a trace of every step, validation of every action in code, and human approval for any action that changes the system.

## What an Agent Is

### An Example First

An alert fires on the bike-share forecasting system in this module's worked example: *api-gateway error rate above 20% for 10 minutes.* The on-call agent gets three things: the alert as its task, the team's runbook in its system prompt, and seven read-only tools that are this module's checks: `list_components`, `get_health`, `get_metric_history`, `run_drift_test`, `run_canary`, `get_deploy_history` and `get_recent_logs`. Here is one run of `1_agent_loop.py`, abridged:

| Step | The model asks for | What comes back | What the model makes of it |
|---|---|---|---|
| 1 | `list_components`, then `get_health` on data-service, both models and the ensemble | The call chain: api-gateway → ensemble → both models → data-service. data-service is down, with memory at 99% and 7 restarts in the hour. The other three have errors but pass their probes, and all show the same 2,050 ms p95 latency | The others are waiting on data-service, so it is the suspect |
| 2 | `get_deploy_history`, `get_recent_logs` and `get_metric_history(memory_pct)` for data-service | The last deploy was eight days ago. The logs show `OOMKilled (memory limit 2Gi)` and then `CrashLoopBackOff`. Memory sat near 57% until four hours ago, then climbed 78 → 99% | It ran out of memory; a bad deploy did not cause this |
| 3 | No tool | | **Answer:** the root cause is data-service out of memory. Restart it and raise its memory limit or add replicas. Do not roll back. Next, check its request-rate history to find out why memory climbed |

The run took 3 model calls and 8 tool calls, about 9,200 input tokens and 4 cents.

Nobody wrote that sequence in code. The alert named api-gateway, but the model never checked it. It read the call chain, saw that everything above data-service was failing in the same way, and went to the bottom of the chain. It checked for a deploy because the runbook says to, and dropped the idea when the dates did not match. It also said what it had not checked. A different alert would take it down a different path with the same code, prompt and tools.

### The General Definition

Take away the incident and every agent has the same shape, a loop around a model:

```
task + tools ──► model ──► asks for a tool? ──yes──► your code runs it ──► result ──┐
                   ▲                │                                              │
                   │                no: answer                                     │
                   └───────────────────────────────────────────────────────────────┘
```

1. Send the task, a system prompt and the tool definitions to the model.
2. If the reply asks for tools, run them in your code and send the results back.
3. Repeat until the reply asks for no tool, or a limit is reached.

In the example, the task is the alert, the tools are the seven checks, and `OpsSystem.execute` in `ops_system.py` is "your code runs it." The loop ended at step 3 because the reply asked for no tool. If it had not, `MAX_STEPS = 12` would have stopped it.

Four facts follow from the loop, and most of the production work comes from them:

- **The model never runs anything.** It returns a tool name and JSON arguments; your code decides what that does and what it may touch. The tool layer is the security boundary.
- **A tool is a name, a description and an input schema.** The model chooses tools from the descriptions, so a description should say *when* to call the tool, not only what it does.
- **The loop must end.** It needs a step limit and a budget, not only the model's own decision to stop.
- **Every step re-sends the whole conversation.** Cost and latency grow with the number of steps. Prompt caching reduces the cost of the repeated part but does not remove it.

## Workflow, Router or Agent

| Design | Who chooses the next step | Example in a team system | Strength | Weakness |
|---|---|---|---|---|
| Ensemble | Fixed math in code | Average two regressors; stack with a meta-model | Predictable, cheap, easy to test | Only combines outputs to the same question |
| Workflow | Code, in a fixed order; a model may fill in steps | Extract fields with a language model, then call the matching member's model | Same path every time; testable | Cannot adapt when a case needs a different path |
| Router | A rule or a model picks one path per request | A classifier sends each request to one member's model | Handles mixed requests at one model call's cost | One wrong route and the answer is wrong |
| Agent | The model, at every step | An on-call assistant that picks checks until it finds a cause | Open-ended tasks whose steps cannot be listed in advance | Variable cost and latency; hardest to test and monitor |

Start with the simplest design that works, and move down the table only when you can show the gain. An agent fits when all four of these hold:

1. **Complexity:** the steps cannot be written down in advance.
2. **Value:** the result is worth the extra cost and latency.
3. **Capability:** the model can do the task reliably. Measure it; do not assume it.
4. **Recoverable errors:** a mistake is caught before it does harm, by tests, review, approval or rollback.

If any of the four fails, use a workflow.

## Common Patterns

- **Tool-use loop (ReAct).** Reason, act, observe, repeat: the loop above. Most production agents are this pattern with good tools.
- **Planner–executor (orchestrator–workers).** One model breaks a task into parts and hands each to a worker, often in parallel. Useful when the parts are independent; the cost multiplies with the number of workers.
- **Evaluator–optimizer.** One model drafts, a second checks the draft against stated criteria, and the loop repeats until the check passes or a limit is reached.
- **Human in the loop.** The agent investigates and proposes; a person approves anything that changes the system.
- **Multi-agent.** Several agents with separate instructions and contexts. Worth it when the work divides cleanly; otherwise one agent with well-designed tools does better.

## What Changes in Production

1. **Least privilege.** Give read-only tools by default. Put every action that changes production (rollback, restart, write, send) behind validation in code and a person's approval. A dedicated tool such as `rollback_deployment(component, version)` can be checked and gated; a general shell command cannot.
2. **Tool output is untrusted data.** Logs, documents, web pages and user fields can contain text that reads like an instruction. This is *prompt injection* (OWASP LLM01). Say in the system prompt that tool output is data, and do not rely on that alone: the validation and approval in point 1 must stop a harmful action even if the model is fooled (OWASP LLM06, Excessive Agency).
3. **Budgets.** Set limits on steps, tool calls, tokens or dollars, and time per run. When a limit is hit, stop and report. Do not let the agent continue unattended (OWASP LLM10, Unbounded Consumption).
4. **Tracing.** Log each run as a sequence of steps: every model call (tokens, latency, stop reason) and every tool call (arguments, result, error, latency), under one trace ID. Without a trace you cannot debug a run, explain a decision or monitor the agent. The OpenTelemetry generative-AI conventions give standard names for these spans.
5. **Monitoring the agent.** An agent is a component and gets the same treatment as your models in System Monitoring: measures, thresholds and alerts. Useful measures are success rate on a golden set, cost per run, steps per run, latency, tool error rate, runs stopped by a budget, and actions a person denied. A change in how often each tool is called is the agent's version of drift. It often follows a change of model, prompt or data.
6. **Evaluation.** The same input can take a different path on each run. Run every test case several times, and report the success rate and the spread of cost and steps, not a single run.
7. **Versioning and rollback.** Pin the model version. Keep the system prompt and tool definitions in the repository with the code, and deploy them like any other change, with a rolling update and a way back.

## How It Connects to This Module's Work

- **Ensemble Approach.** "Routing between models" is one of the accepted methods, and a rule or a small classifier is usually enough. If your team uses a language model to route, or an agent to integrate the members' models, Part 3 compares it with the best single model *and* with a fixed rule. Report its cost per request and latency next to the metric.
- **System Monitoring.** If your system includes a language-model step or an agent, add its measures (cost per request, steps, tool error rate, latency) to the Part 1 and Part 3 tables, and its traces to your logging.
- No team is required to build an agent. The point is to know when one earns its place, and what it costs when it does.

## Worked Example

The [agents repository](https://github.com/AI-Demonstrations/agents) holds three short examples built around one task: an on-call agent that diagnoses incidents in a small simulated system with a data service, two models, an ensemble and an API. Its tools are this module's checks: health, metric history, Kolmogorov-Smirnov, PSI and chi-square drift tests, canary inputs, deploy history and logs.

| File | What it shows |
|---|---|
| `1_agent_loop.py` | The agent loop written out in full, printing every step (the run in *An Example First*) |
| `2_workflow_vs_agent.py` | A fixed workflow against the agent on five incidents with known causes: accuracy, model calls, tokens, dollars and seconds |
| `3_guarded_agent.py` | The agent with production controls: trace, budgets, validation, human approval, tool failures, an injected instruction in a log line, and a report that monitors the agent against thresholds |

Everything runs on your VM's CPU. The examples call the Anthropic API and need an `ANTHROPIC_API_KEY` in a `.env` file, as the language-model sections of the Module 11 explainability examples did. A run costs a few cents.

## Ensemble or Agent: How to Decide

An ensemble and an agent do different kinds of work, so the job usually decides between them. An ensemble combines models to make one better prediction. An agent works out a sequence of steps to complete a task. Look at the job against each of these factors:

| Factor | Points to an ensemble | Points to an agent |
|---|---|---|
| **The job** | One prediction per request, made the same way every time | A task whose next step depends on what the last step found |
| **Input and output** | Structured features in; a number, class or probability out | An open request in, such as an alert or a question; a finding, a plan or an action out |
| **Volume and latency** | Many requests, each answered in milliseconds | Few requests, where seconds or minutes are acceptable |
| **Cost per request** | Fixed, and close to zero once the models are trained | Cents per run and variable, because each step is a model call |
| **How you show it works** | A metric on a held-out test set, such as MAE or AUC | Success rate over repeated runs on cases with known answers |
| **What you need to build it** | Labeled training data | Tools, written instructions such as a runbook, and test cases |
| **How it fails** | A wrong number, with an error you can measure | A wrong path or a wrong action, which needs budgets, validation and approval |
| **How you explain a result** | Feature importance and SHAP, as in Module 11 | The trace of the steps the model took |
| **What drifts** | Input data and predictions; you retrain | The model version, the prompt and the mix of tool calls; you re-test |

If most rows point to an ensemble, use one. If most point to an agent, try a fixed workflow first and check the four conditions in *Workflow, Router or Agent*. The agent has to beat the workflow on a measure you chose before you ran either.

The two can work together. An agent can call an ensemble as one of its tools, and an ensemble can be one of the components an agent monitors.

### Use Case: Two Jobs in the Bike-Share System

The system in the worked example has two jobs, and they land on opposite sides of the table.

**Job 1: forecast each station for the next hour.** Each station needs a forecast of its rides and of its chance of running out of bikes. Both models score the same station from the same features, and the `ensemble` component combines their outputs into one forecast. Every row points to an ensemble. The steps never change, requests are many and must be answered quickly, and actual rides arrive an hour later, so the team has labels to train on and MAE to measure. An agent would add seconds and cents to every forecast, where the ensemble takes milliseconds and a fraction of a cent. It would gain nothing for that cost, because it would have no choices to make.

**Job 2: find out why the forecast broke.** When an alert fires, someone has to find the cause, and the right checks depend on what each one shows. An out-of-memory crash, a bad deploy and drifted inputs each need different evidence, as *An Example First* showed. Alerts are rare, and an answer in a minute or two is fast enough. Mistakes can be caught, because the agent's tools only read and a person approves any change (`3_guarded_agent.py`). Incidents are too few and too varied to train a model on, but past incidents with known causes make a test set. Most rows point to an agent. Before you build one, `2_workflow_vs_agent.py` runs the comparison above: a fixed workflow that runs every check against the agent, on five incidents with known causes. The agent earns its place only if it gets more of them right, at a cost per run the team accepts.

## Readings

**Core**

- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (2024): workflows against agents, the common patterns, and when not to build an agent
- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): defining tools, the `tool_use` / `tool_result` round trip, and strict tool use
- OWASP, [Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/): read LLM01 Prompt Injection, LLM06 Excessive Agency and LLM10 Unbounded Consumption

**Supporting**

- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023: the paper behind the tool-use loop
- OpenTelemetry, [Semantic conventions for generative AI](https://github.com/open-telemetry/semantic-conventions-genai): standard span names and attributes for model, agent and tool calls
- [Model Context Protocol](https://modelcontextprotocol.io/): an open standard for exposing tools and data to agents
