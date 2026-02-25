"""Report Agent — synthesizes data findings into a structured markdown report."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic_ai import Agent, RunContext

from agents.models import DataDeps, ReportResult
from data.loader import get_df

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """\
You are a data analyst writing concise, professional reports.
Use your tools to gather statistics and findings, then synthesize them into a
clear 1-2 page markdown report. Structure the report with sections:
Executive Summary, Key Findings, Top Performers, and Recommendations.
Be specific with numbers. Save the report using write_report_to_file.

Return a structured ReportResult with:
- file_path: the path returned by write_report_to_file
- highlights: top 3-5 bullet-point findings from the report
"""

report_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=DataDeps,
    output_type=ReportResult,
    system_prompt=SYSTEM_PROMPT,
)


@report_agent.tool
def get_summary_stats(ctx: RunContext[DataDeps]) -> str:
    """Return global summary statistics for intervention_effect and intervention_cost."""
    df = ctx.deps.df
    stats = df[["intervention_effect", "intervention_cost"]].describe()
    per_action = (
        df.groupby("action")[["intervention_effect", "intervention_cost"]]
        .mean()
        .round(2)
    )
    per_scenario = (
        df.groupby("scenario")[["intervention_effect", "intervention_cost"]]
        .mean()
        .round(2)
    )
    return (
        f"=== Global Stats ===\n{stats.to_string()}\n\n"
        f"=== Mean by Action ===\n{per_action.to_string()}\n\n"
        f"=== Mean by Scenario ===\n{per_scenario.to_string()}"
    )


@report_agent.tool
def get_top_performers(ctx: RunContext[DataDeps], n: int = 10) -> str:
    """Return the top N entities ranked by mean intervention_effect."""
    df = ctx.deps.df
    ranked = (
        df.groupby("entity")
        .agg(
            mean_effect=("intervention_effect", "mean"),
            max_effect=("intervention_effect", "max"),
            mean_cost=("intervention_cost", "mean"),
            mean_efficiency=(
                "intervention_effect",
                lambda x: (x / df.loc[x.index, "intervention_cost"]).mean(),
            ),
        )
        .reset_index()
        .sort_values("mean_effect", ascending=False)
        .head(n)
    )
    return ranked.to_string(index=False)


@report_agent.tool
def get_bottom_performers(ctx: RunContext[DataDeps], n: int = 5) -> str:
    """Return the bottom N entities by mean intervention_effect."""
    df = ctx.deps.df
    ranked = (
        df.groupby("entity")
        .agg(mean_effect=("intervention_effect", "mean"))
        .reset_index()
        .sort_values("mean_effect", ascending=True)
        .head(n)
    )
    return ranked.to_string(index=False)


@report_agent.tool
def get_best_action_per_entity(ctx: RunContext[DataDeps]) -> str:
    """
    For each entity, return the action with the highest mean effect.
    Returns a summary table showing which action wins most often.
    """
    df = ctx.deps.df
    best = (
        df.groupby(["entity", "action"])["intervention_effect"]
        .mean()
        .reset_index()
    )
    idx = best.groupby("entity")["intervention_effect"].idxmax()
    best_per_entity = best.loc[idx][["entity", "action", "intervention_effect"]]
    action_counts = best_per_entity["action"].value_counts()
    return (
        f"Best action frequency across all entities:\n{action_counts.to_string()}\n\n"
        f"Sample (first 10):\n{best_per_entity.head(10).to_string(index=False)}"
    )


@report_agent.tool
def write_report_to_file(ctx: RunContext[DataDeps], content: str) -> str:
    """Save the report markdown content to the reports/ directory. Returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"report_{timestamp}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def get_report_agent() -> tuple[Agent, DataDeps]:
    return report_agent, DataDeps(df=get_df())
