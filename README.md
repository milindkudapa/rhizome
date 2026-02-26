# Multi-Agent Analysis System

A multi-agent AI system built with [PydanticAI](https://ai.pydantic.dev/) that answers natural language queries about intervention data. An orchestrator routes each query to specialised sub-agents for analysis, visualisation, and reporting — all backed by typed Pydantic models end-to-end.

---

## Architecture

```
User query (CLI: main.py)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Orchestrator  (claude-sonnet-4-6)                  │
│  output: OrchestratorResponse                       │
│                                                     │
│  tools:                                             │
│    run_analyze   ──▶  Analyze Agent                 │
│    run_visualize ──▶  Visualize Agent               │
│    run_report    ──▶  Report Agent                  │
└─────────────────────────────────────────────────────┘
```

**Routing rules (enforced in the orchestrator's system prompt):**

| Request type | Orchestrator behaviour |
|---|---|
| Pure analysis | `run_analyze`  |
| Visualization | `run_visualize`  |
| Report | `run_analyze("Give me a dataset overview")` → `run_report` |
| Combined ("analyze and plot") | `run_analyze` → `run_visualize` |

---

## Pydantic models

All agent deps and LLM outputs are typed Pydantic models (`pydantic.BaseModel`). The LLM is constrained to return validated JSON matching each schema via PydanticAI's `output_type`.

### Shared deps

| Model | Field | Description |
|---|---|---|
| `DataDeps` | `df: pd.DataFrame` | Singleton DataFrame injected into every agent |

### Agent output types

| Model | Fields | Returned by |
|---|---|---|
| `AnalysisResult` | `answer`, `recommended_action`, `key_findings` | Analyze Agent |
| `VisualizationResult` | `plot_paths`, `description` | Visualize Agent |
| `ReportResult` | `file_path`, `highlights` | Report Agent |
| `OrchestratorResponse` | `answer`, `plot_paths`, `report_path` | Orchestrator |

---

## Agents and tools

### Orchestrator (`agents/orchestrator.py`)
Routes queries and composes the final response. Calls sub-agents as tools and passes their JSON-serialised output back into its own reasoning.

| Tool | Delegates to |
|---|---|
| `run_analyze(query)` | Analyze Agent |
| `run_visualize(query)` | Visualize Agent |
| `run_report(query)` | Report Agent |

### Analyze Agent (`agents/analyze.py`)
Answers quantitative questions about the dataset using pandas operations.

| Tool | What it does |
|---|---|
| `query_entity_data(entity_id)` | All rows for an entity as JSON |
| `get_best_scenario(entity_id, action)` | Scenario with highest effect for a given action |
| `get_highest_effect_scenario_across_actions(entity_id)` | Global best (action, scenario) for an entity |
| `recommend_action(entity_id)` | Best action by mean cost-efficiency (effect / cost) |
| `compute_cost_efficiency(entity_id)` | Full efficiency table sorted descending |
| `compare_entities(entity_ids)` | Side-by-side mean effect / cost across multiple entities |
| `get_dataset_overview()` | Compact global summary — aggregated stats, per-action/scenario means, top-5/bottom-5 entities in a single call |

### Visualize Agent (`agents/visualize.py`)
Generates matplotlib/seaborn charts and saves them as PNG files to `plots/`.

| Tool | Output |
|---|---|
| `plot_entity_bar_chart(entity_id)` | Effect per (action, scenario) bar chart |
| `plot_cost_vs_effect_scatter(entity_id)` | Cost vs effect scatter, labelled by action/scenario |
| `plot_global_heatmap()` | Entity × action heatmap of mean effect across all 99 entities |

### Report Agent (`agents/report.py`)
Synthesises dataset statistics into a structured markdown report saved to `reports/`.

| Tool | What it does |
|---|---|
| `get_summary_stats()` | Global and per-action/scenario descriptive statistics |
| `get_top_performers(n)` | Top N entities by mean effect (includes max, mean cost, efficiency) |
| `get_bottom_performers(n)` | Bottom N entities by mean effect |
| `get_best_action_per_entity()` | Which action wins most often across all 99 entities |
| `write_report_to_file(content)` | Saves markdown to `reports/report_<timestamp>.md` |

---

## Dataset

`synthetic_data.csv` — 891 rows (99 entities × 3 actions × 3 scenarios).

| Column | Values |
|---|---|
| `entity` | `entity_1` … `entity_99` |
| `action` | `action_1`, `action_2`, `action_3` |
| `scenario` | `scenario_1`, `scenario_2`, `scenario_3` |
| `intervention_effect` | float — magnitude of the intervention effect |
| `intervention_cost` | float — cost |

---

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/) · Python 3.11+

```bash
# 1. Install dependencies
uv sync

# 2. Add your Anthropic API key
cp .env.example .env
# then edit .env:  ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the interactive CLI
uv run python main.py
```

---

## CLI

`main.py` is an async REPL that maintains conversation history across turns. On start it prints example queries, then loops until the user types `quit` / `exit` / `q`.

Output is structured — plots and report paths are printed separately from the text answer:

```
Agent: For entity_1, action_2 has the best cost-efficiency ...

Plots saved:
  • /path/to/plots/entity_1_scatter.png

Report saved: /path/to/reports/report_20260224_143021.md
```

### Example queries

| Intent | Query |
|---|---|
| Analyze | `For entity_42, which scenario yields the highest effect?` |
| Analyze | `What action should we consider for entity_1?` |
| Analyze | `Compare entity_5 and entity_12 across all actions` |
| Analyze | `Which entity has the best cost-efficiency overall?` |
| Visualize | `Plot entity_7's cost vs effect across all scenarios` |
| Visualize | `Show me a bar chart for entity_20` |
| Visualize | `Generate a global heatmap of all entities` |
| Report | `Generate a 1-2 page report summarizing the dataset` |
| Report | `What are the top 10 entities by mean intervention effect?` |

---

## Tests

```bash
# Unit tests — no API key needed (~3s)
uv run pytest -m "not integration" -v

# Integration tests — 1 real LLM call per agent (~55s, uses API credits)
uv run pytest tests/test_integration.py -v

# All tests
uv run pytest -v
```

### Test layout

```
tests/
├── conftest.py              # shared fixtures, get_tool_fn helper, .env loader
├── test_loader.py           # data integrity: shape, columns, types, singleton
├── test_models.py           # Pydantic validation: required fields, defaults, serialisation
├── test_analyze_tools.py    # tool logic: correct pandas results, edge cases
├── test_visualize_tools.py  # file creation: PNG exists, correct name, non-empty
├── test_report_tools.py     # file creation + stat correctness
└── test_integration.py      # one real LLM call per agent, output type checks
```

Unit tests call tool functions directly (via `agent._function_toolset.tools[name].function`) using a `SimpleNamespace` mock context — no LLM calls, no network.

---

## Project layout

```
rhizome/
├── main.py                  # async CLI REPL
├── agents/
│   ├── models.py            # DataDeps + all Pydantic output models
│   ├── analyze.py           # Analyze Agent + 7 tools
│   ├── visualize.py         # Visualize Agent + 3 tools
│   ├── report.py            # Report Agent + 5 tools
│   └── orchestrator.py      # Orchestrator Agent + 3 delegation tools
├── data/
│   └── loader.py            # singleton pd.DataFrame loader
├── tests/                   # pytest suite (91 unit + 4 integration)
├── plots/                   # generated PNG files (gitignored)
├── reports/                 # generated markdown reports (gitignored)
├── synthetic_data.csv       # 891-row intervention dataset
├── pyproject.toml           # uv-managed deps + pytest config
└── .env.example             # ANTHROPIC_API_KEY template
```
