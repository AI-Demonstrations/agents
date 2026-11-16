# Agents: When the Model Chooses the Next Step

An ensemble combines models in a way you fix in code: average these outputs, stack those, route by this rule. An **agent** hands that choice to a language model. The model is given a task and a set of tools, functions your code runs, which can include your team's models and your monitoring checks. It decides which tool to call next from what the last one returned, and stops when it can answer. That makes an agent the most flexible way to integrate models and checks, and also the hardest to predict, test and pay for. This section covers what an agent is in code, when it is worth its cost over a fixed pipeline, and what it needs before it may change a production system.

## Objective

12.5 Decide whether a fixed workflow or an agent should integrate a set of models or checks, and measure the choice. Name the controls an agent needs in production: step and cost budgets, a trace of every step, validation of every action in code, and human approval for any action that changes the system.

## What an Agent Is

An agent is a loop around a model:

```
task + tools ──► model ──► asks for a tool? ──yes──► your code runs it ──► result ──┐
                   ▲                │                                              │
                   │                no: answer                                     │
                   └───────────────────────────────────────────────────────────────┘
```

1. Send the task, a system prompt and the tool definitions to the model.
2. If the reply asks for tools, run them in your code and send the results back.
3. Repeat until the reply asks for no tool, or a limit is reached.

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
| `1_agent_loop.py` | The agent loop written out in full, printing every step |
| `2_workflow_vs_agent.py` | A fixed workflow against the agent on five incidents with known causes: accuracy, model calls, tokens, dollars and seconds |
| `3_guarded_agent.py` | The agent with production controls: trace, budgets, validation, human approval, tool failures, an injected instruction in a log line, and a report that monitors the agent against thresholds |

Everything runs on your VM's CPU. The examples call the Anthropic API and need an `ANTHROPIC_API_KEY` in a `.env` file, as the language-model sections of the Module 11 explainability examples did. A run costs a few cents.

## Readings

**Core**

- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (2024): workflows against agents, the common patterns, and when not to build an agent
- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): defining tools, the `tool_use` / `tool_result` round trip, and strict tool use
- OWASP, [Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/): read LLM01 Prompt Injection, LLM06 Excessive Agency and LLM10 Unbounded Consumption

**Supporting**

- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023: the paper behind the tool-use loop
- OpenTelemetry, [Semantic conventions for generative AI](https://github.com/open-telemetry/semantic-conventions-genai): standard span names and attributes for model, agent and tool calls
- [Model Context Protocol](https://modelcontextprotocol.io/): an open standard for exposing tools and data to agents
