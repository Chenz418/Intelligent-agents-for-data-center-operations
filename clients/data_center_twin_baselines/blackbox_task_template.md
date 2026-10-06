# Data Center Twin Task

Task ID: {{AGENT_TASK_ID}}

You are solving one Data Center Twin benchmark task in an isolated workspace.
Use only the files in this directory and the neutral tool below. Do not inspect
or rely on repository source files, scenario manifests, evaluator code, fault
injection code, expected answers, success criteria, or benchmark registries.

## Initial Observation

Read `INITIAL_OBSERVATION.json` before taking an action. It contains the current
agent-visible causal snapshot and already counts as visible evidence. Do not call
`observe` merely to retrieve that same snapshot again; use it after state changes
or when you need a newer causal cut or deliberate drill-down.

## Tool

Use `dc_twin_tool.py` for all benchmark interaction:

```text
python3 dc_twin_tool.py action-space
python3 dc_twin_tool.py observe --log-limit 20 --include-config
python3 dc_twin_tool.py action --json '{"action_type": "<visible-action-type>", "parameters": {"target": "<visible-target-from-observation>"}, "advance_ticks": 1}'
python3 dc_twin_tool.py submit --json '{"incident_detected": true, "diagnosis": "<concise-free-form-underlying-mechanism>", "target": "<visible-target-from-observation>", "evidence": ["<agent-visible-supporting-evidence>"]}'
python3 dc_twin_tool.py submit-empty
```

The tool prints JSON responses. Submit once when you have a final answer.
Do not print `submit(...)`, `dc_twin_observe(...)`, or other markdown API-call
blocks as your final response. The evaluator only records actions executed via
`python3 dc_twin_tool.py ...`.

## Visible Benchmark Context

```json
{{CONTEXT_JSON}}
```
