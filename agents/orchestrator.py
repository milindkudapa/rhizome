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

**Visualization always requires prior analysis.**
If the user asks for a plot or chart — even if they did NOT explicitly ask for
analysis — you MUST call run_analyze first to retrieve the relevant numbers and
context for that entity/action/scenario. Then pass both the original request AND
the analysis findings into run_visualize. This ensures every chart is grounded in
concrete data rather than producing a generic or empty plot.

Example: user says "plot entity_7's cost vs effect"
→ Step 1: run_analyze("Get all intervention_effect and intervention_cost values
  for entity_7 across all actions and scenarios")
→ Step 2: run_visualize("Plot cost vs effect scatter for entity_7. Analysis
  context: [paste step 1 result here]")

**Reports benefit from analysis too.**
If the user asks for a report, call run_analyze for summary stats first, then
pass those findings to run_report.

**Pure analysis queries** only need run_analyze.

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
    Also call this BEFORE run_visualize to gather data context for a plot.
    Returns a JSON-serialised AnalysisResult.
    """
    result = await analyze_agent.run(query, deps=DataDeps(df=ctx.deps.df))
    return result.output.model_dump_json()


@orchestrator.tool
async def run_visualize(ctx: RunContext[DataDeps], query: str) -> str:
    """
    Delegate a visualization request to the Visualize Agent.
    The query should include both what to plot AND the analysis context obtained
    from run_analyze. Returns a JSON-serialised VisualizationResult with plot_paths.
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
