"""Orchestrator Agent — routes user queries to Analyze, Visualize, or Report agents."""

from pydantic_ai import Agent, RunContext

from agents.analyze import analyze_agent
from agents.models import DataDeps, OrchestratorResponse
from agents.report import report_agent
from agents.visualize import visualize_agent
from data.loader import get_df

SYSTEM_PROMPT = """\
You are the orchestrator for a multi-agent data analysis system. You have access to
three specialized agents:

1. **run_analyze** — answers data questions (effects, costs, rankings, comparisons)
2. **run_visualize** — generates charts and plots saved as PNG files
3. **run_report** — creates a written 1-2 page summary report

## Routing rules

### Visualization requests (any chart or plot)
Call run_visualize directly. The visualize agent has full DataFrame access and
its plotting tools read from the data themselves — no prior analysis step needed.

### Reports
Call run_analyze("Give me a dataset overview") first, then run_report with the
findings. The overview tool returns aggregated stats in a single call.

### Pure analysis queries
Only run_analyze is needed.

### Combined requests (e.g. "analyze and plot entity_5")
Call run_analyze first, then run_visualize.

Return a structured OrchestratorResponse with:
- answer: full response to relay to the user incorporating all agent findings
- plot_paths: list of any PNG file paths produced (empty list if none)
- report_path: path to any generated report file, or null

The dataset covers 99 entities (entity_1 … entity_99), 3 actions (action_1/2/3),
and 3 scenarios (scenario_1/2/3), with intervention_effect and intervention_cost columns.
"""

orchestrator = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=DataDeps,
    output_type=OrchestratorResponse,
    system_prompt=SYSTEM_PROMPT,
)


@orchestrator.tool
async def run_analyze(ctx: RunContext[DataDeps], query: str) -> str:
    """
    Delegate a data analysis query to the Analyze Agent.
    Use for questions about effects, costs, rankings, recommendations, or comparisons.
    For a full dataset summary pass "Give me a dataset overview" — the agent uses
    get_dataset_overview() which returns aggregated stats in a single compact call.
    Returns a JSON-serialised AnalysisResult.
    """
    result = await analyze_agent.run(query, deps=DataDeps(df=ctx.deps.df))
    return result.output.model_dump_json()


@orchestrator.tool
async def run_visualize(ctx: RunContext[DataDeps], query: str) -> str:
    """
    Delegate a visualization request to the Visualize Agent.
    The agent has direct DataFrame access — just describe what to plot.
    Returns a JSON-serialised VisualizationResult with plot_paths.
    """
    result = await visualize_agent.run(query, deps=DataDeps(df=ctx.deps.df))
    return result.output.model_dump_json()


@orchestrator.tool
async def run_report(ctx: RunContext[DataDeps], query: str) -> str:
    """
    Delegate a reporting request to the Report Agent.
    Use when the user asks for a written report, summary, or executive overview.
    Include any analysis context already gathered. Returns a JSON-serialised ReportResult.
    """
    result = await report_agent.run(query, deps=DataDeps(df=ctx.deps.df))
    return result.output.model_dump_json()


def get_orchestrator() -> tuple[Agent, DataDeps]:
    return orchestrator, DataDeps(df=get_df())
