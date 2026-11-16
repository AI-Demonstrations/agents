"""Demo 2: a fixed workflow against an agent, on the same incidents.

Before choosing an agent, measure it against the simpler design. Here both
designs diagnose each scenario in ops_system.py and return the same structured
diagnosis, which is graded against the scenario's known root cause:

  workflow  Code runs every check in a fixed order (all health, deploys, logs,
            canaries, drift tests and the key metric histories), then makes
            one model call that reads them all and diagnoses.
  agent     The model chooses which checks to run, one step at a time (the
            loop from demo 1, as agent_common.run_agent).

The table reports, per scenario and design: whether the diagnosis was right,
model calls, tool calls, tokens, dollars and seconds. It is the same comparison
the Ensemble Approach assignment asks for: the combined system against the
simplest thing that works, on one metric everyone agreed on beforehand.

    python 2_workflow_vs_agent.py                         # every scenario, once
    python 2_workflow_vs_agent.py --repeats 3             # agents vary between runs; measure it
    python 2_workflow_vs_agent.py --scenarios bad_deploy noise

Results are saved to runs/2_workflow_vs_agent.json.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from agent_common import EFFORT, MODEL, RUNS, Usage, call_claude, make_client, run_agent, save_json, text_of
from ops_system import COMPONENTS, DRIFT_FEATURES, MODEL_COMPONENTS, READ_TOOLS, SCENARIOS, OpsSystem, \
    system_prompt

# Both designs must answer in this shape, so they can be graded the same way.
DIAGNOSIS = {"type": "json_schema", "schema": {
    "type": "object",
    "properties": {
        "root_cause_component": {"type": "string", "enum": ["none", *COMPONENTS]},
        "cause": {"type": "string", "enum": ["none", "data_drift", "bad_deploy", "outage"]},
        "action": {"type": "string", "enum": ["none", "rollback", "restart_or_scale"]},
        "evidence": {"type": "string", "description": "Two or three sentences citing the checks that decided it"},
    },
    "required": ["root_cause_component", "cause", "action", "evidence"],
    "additionalProperties": False,
}}
GRADED = ["root_cause_component", "cause", "action"]


def task_for(ops):
    return f"{ops.trigger()}\n\nDiagnose the root cause and choose the action."


def gather_everything(ops):
    """The workflow's fixed plan: every check a monitoring dashboard would show."""
    checks = {"components": ops.list_components()}
    for c in COMPONENTS:
        checks[f"health {c}"] = ops.get_health(c)
        checks[f"deploys {c}"] = ops.get_deploy_history(c)
        checks[f"logs {c}"] = ops.get_recent_logs(c)
        checks[f"error_rate_pct history {c}"] = ops.get_metric_history(c, "error_rate_pct")
    for m in MODEL_COMPONENTS:
        checks[f"canary {m}"] = ops.run_canary(m)
        for f in DRIFT_FEATURES:
            checks[f"drift {m} {f}"] = ops.drift_test(m, f)
    for metric in ("mae", "prediction_mean"):
        checks[f"{metric} history ensemble"] = ops.get_metric_history("ensemble", metric)
    return checks


def run_workflow(client, ops):
    start = time.perf_counter()
    checks = gather_everything(ops)
    prompt = (f"{task_for(ops)}\n\nThe results of every standard check follow, as JSON.\n\n"
              f"<checks>\n{json.dumps(checks, indent=1)}\n</checks>")
    response, _ = call_claude(client, system=system_prompt(), messages=[{"role": "user", "content": prompt}],
                              output_format=DIAGNOSIS)
    return dict(answer=text_of(response), stop=response.stop_reason, llm_calls=1, tool_calls=len(checks),
                tool_errors=0, usage=Usage.of(response.usage), seconds=time.perf_counter() - start)


def run_agent_design(client, ops):
    r = run_agent(client, system=system_prompt(), task=task_for(ops), tools=READ_TOOLS,
                  execute_tool=ops.execute, max_steps=15, output_format=DIAGNOSIS)
    return dict(answer=r.answer, stop=r.stop, llm_calls=r.steps, tool_calls=r.tool_calls,
                tool_errors=r.tool_errors, usage=r.usage, seconds=r.seconds)


def grade(answer, truth):
    try:
        diagnosis = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return None, False
    return diagnosis, all(diagnosis.get(k) == truth[k] for k in GRADED)


def one_run(client, scenario, design, repeat):
    ops = OpsSystem(scenario)
    out = (run_workflow if design == "workflow" else run_agent_design)(client, ops)
    diagnosis, correct = grade(out["answer"], SCENARIOS[scenario]["truth"])
    return dict(scenario=scenario, design=design, repeat=repeat, correct=correct, diagnosis=diagnosis,
                stop=out["stop"], llm_calls=out["llm_calls"], tool_calls=out["tool_calls"],
                tool_errors=out["tool_errors"], usage=out["usage"].as_dict(), seconds=round(out["seconds"], 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="runs in parallel (each run is timed on its own)")
    args = parser.parse_args()

    client = make_client()
    jobs = [(s, d, r) for r in range(args.repeats) for s in args.scenarios for d in ("workflow", "agent")]
    print(f"Model {MODEL}, effort {EFFORT}: {len(jobs)} runs, {args.workers} at a time.\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda job: one_run(client, *job), jobs))

    header = f"{'scenario':16} {'design':9} {'diagnosis (component / cause / action)':44} {'ok':3} " \
             f"{'LLM':>4} {'tools':>5} {'tok in':>7} {'tok out':>7} {'$':>6} {'sec':>5}"
    print(header)
    print("-" * len(header))
    for row in rows:
        d = row["diagnosis"]
        said = " / ".join(d[k] for k in GRADED) if d else f"(no diagnosis: {row['stop']})"
        u = row["usage"]
        print(f"{row['scenario']:16} {row['design']:9} {said:44} {'yes' if row['correct'] else 'NO':3} "
              f"{row['llm_calls']:>4} {row['tool_calls']:>5} {u['input'] + u['cache_read'] + u['cache_write']:>7} "
              f"{u['output']:>7} {u['cost_usd']:>6.3f} {row['seconds']:>5.1f}")

    print("\nBy design")
    for design in ("workflow", "agent"):
        sel = [r for r in rows if r["design"] == design]
        n = len(sel)
        print(f"  {design:9} correct {sum(r['correct'] for r in sel)}/{n}, "
              f"mean ${sum(r['usage']['cost_usd'] for r in sel) / n:.3f} and "
              f"{sum(r['seconds'] for r in sel) / n:.1f} s per incident, "
              f"mean {sum(r['llm_calls'] for r in sel) / n:.1f} model calls and "
              f"{sum(r['tool_calls'] for r in sel) / n:.1f} checks")
    print("\nThe workflow always pays for every check; the agent pays per step it chooses. Which wins depends on\n"
          "how many checks exist and how many an incident needs: try adding components or checks and re-run.")

    path = save_json(RUNS / "2_workflow_vs_agent.json", {"model": MODEL, "effort": EFFORT, "runs": rows})
    print(f"\nResults: {path.relative_to(RUNS.parent)}")


if __name__ == "__main__":
    main()
