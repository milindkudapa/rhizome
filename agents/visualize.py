"""Visualize Agent — generates plots from the intervention dataset."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic_ai import Agent, RunContext

from agents.models import DataDeps, VisualizationResult
from data.loader import get_df

matplotlib.use("Agg")

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """\
You are a data visualization expert. Use your tools to create charts from the
intervention dataset. After calling each plotting tool, note the returned file path.

Return a structured VisualizationResult with:
- plot_paths: list of every file path returned by the plotting tools
- description: what each chart shows and any notable patterns visible in the data
"""

visualize_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=DataDeps,
    output_type=VisualizationResult,
    system_prompt=SYSTEM_PROMPT,
)


@visualize_agent.tool
def plot_entity_bar_chart(ctx: RunContext[DataDeps], entity_id: str) -> str:
    """
    Bar chart of intervention_effect for every (action, scenario) combination
    for a single entity. Saves to plots/ and returns the file path.
    """
    df = ctx.deps.df
    rows = df[df["entity"] == entity_id].copy()
    if rows.empty:
        return f"No data found for entity '{entity_id}'."

    rows["label"] = rows["action"] + "\n" + rows["scenario"]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(rows))
    ax.bar(rows["label"], rows["intervention_effect"], color=colors)
    ax.set_title(f"Intervention Effect — {entity_id}")
    ax.set_xlabel("Action / Scenario")
    ax.set_ylabel("Intervention Effect")
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()

    path = PLOTS_DIR / f"{entity_id}_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


@visualize_agent.tool
def plot_cost_vs_effect_scatter(ctx: RunContext[DataDeps], entity_id: str) -> str:
    """
    Scatter plot of intervention_cost vs intervention_effect for a single entity,
    with each point labeled by action and scenario. Saves to plots/.
    """
    df = ctx.deps.df
    rows = df[df["entity"] == entity_id].copy()
    if rows.empty:
        return f"No data found for entity '{entity_id}'."

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", rows["action"].nunique())
    action_colors = {a: palette[i] for i, a in enumerate(rows["action"].unique())}

    for _, row in rows.iterrows():
        ax.scatter(
            row["intervention_cost"],
            row["intervention_effect"],
            color=action_colors[row["action"]],
            s=100,
            zorder=3,
        )
        ax.annotate(
            f"{row['action']}\n{row['scenario']}",
            (row["intervention_cost"], row["intervention_effect"]),
            fontsize=7,
            textcoords="offset points",
            xytext=(5, 5),
        )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=a
        )
        for a, c in action_colors.items()
    ]
    ax.legend(handles=handles, title="Action")
    ax.set_title(f"Cost vs Effect — {entity_id}")
    ax.set_xlabel("Intervention Cost")
    ax.set_ylabel("Intervention Effect")
    plt.tight_layout()

    path = PLOTS_DIR / f"{entity_id}_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


@visualize_agent.tool
def plot_global_heatmap(ctx: RunContext[DataDeps]) -> str:
    """
    Heatmap of mean intervention_effect across all entities, grouped by action.
    Rows = entities, columns = actions. Saves to plots/.
    """
    df = ctx.deps.df
    pivot = df.pivot_table(
        index="entity",
        columns="action",
        values="intervention_effect",
        aggfunc="mean",
    )
    pivot = pivot.loc[
        sorted(pivot.index, key=lambda x: int(x.split("_")[1]))
    ]

    fig, ax = plt.subplots(figsize=(8, max(10, len(pivot) * 0.22)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Mean Effect"},
    )
    ax.set_title("Mean Intervention Effect by Entity × Action")
    ax.set_xlabel("Action")
    ax.set_ylabel("Entity")
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()

    path = PLOTS_DIR / "global_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def get_visualize_agent() -> tuple[Agent, DataDeps]:
    return visualize_agent, DataDeps(df=get_df())
