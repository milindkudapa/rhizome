"""Analyze Agent — answers data queries about entities, actions, and scenarios."""

from pydantic_ai import Agent, RunContext

from agents.models import AnalysisResult, DataDeps
from data.loader import get_df

SYSTEM_PROMPT = """\
You are a data analyst with access to a dataset of interventions applied to entities.
Each row describes an (entity, action, scenario) combination with an intervention_effect
and an intervention_cost. Use your tools to answer questions precisely.
When recommending actions, consider both effect and cost efficiency (effect / cost).
Always cite the specific numbers you find.

Return your response as a structured AnalysisResult with:
- answer: complete response with exact numbers
- recommended_action: the best action name if a recommendation was requested, else null
- key_findings: up to five short bullet-point facts from the data
"""

analyze_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=DataDeps,
    output_type=AnalysisResult,
    system_prompt=SYSTEM_PROMPT,
)


@analyze_agent.tool
def query_entity_data(ctx: RunContext[DataDeps], entity_id: str) -> str:
    """Return all rows for a given entity as a JSON string."""
    df = ctx.deps.df
    rows = df[df["entity"] == entity_id]
    if rows.empty:
        return f"No data found for entity '{entity_id}'."
    return rows.to_json(orient="records", indent=2)


@analyze_agent.tool
def get_best_scenario(
    ctx: RunContext[DataDeps], entity_id: str, action: str
) -> str:
    """Return the scenario with the highest effect for a given entity and action."""
    df = ctx.deps.df
    rows = df[(df["entity"] == entity_id) & (df["action"] == action)]
    if rows.empty:
        return f"No data found for entity '{entity_id}', action '{action}'."
    best = rows.loc[rows["intervention_effect"].idxmax()]
    return (
        f"For {entity_id} / {action}: best scenario is '{best['scenario']}' "
        f"with effect={best['intervention_effect']:.2f}, cost={best['intervention_cost']:.2f}."
    )


@analyze_agent.tool
def get_highest_effect_scenario_across_actions(
    ctx: RunContext[DataDeps], entity_id: str
) -> str:
    """Return the (action, scenario) combination with the highest effect for an entity."""
    df = ctx.deps.df
    rows = df[df["entity"] == entity_id]
    if rows.empty:
        return f"No data found for entity '{entity_id}'."
    best = rows.loc[rows["intervention_effect"].idxmax()]
    return (
        f"For {entity_id}: highest effect is {best['intervention_effect']:.2f} "
        f"at action='{best['action']}', scenario='{best['scenario']}' "
        f"(cost={best['intervention_cost']:.2f})."
    )


@analyze_agent.tool
def recommend_action(ctx: RunContext[DataDeps], entity_id: str) -> str:
    """Recommend the best action for an entity based on mean cost-efficiency (effect/cost)."""
    df = ctx.deps.df
    rows = df[df["entity"] == entity_id].copy()
    if rows.empty:
        return f"No data found for entity '{entity_id}'."
    rows["efficiency"] = rows["intervention_effect"] / rows["intervention_cost"]
    summary = (
        rows.groupby("action")["efficiency"]
        .mean()
        .reset_index()
        .sort_values("efficiency", ascending=False)
    )
    lines = [f"Cost-efficiency (effect/cost) for {entity_id}:"]
    for _, r in summary.iterrows():
        lines.append(f"  {r['action']}: {r['efficiency']:.6f}")
    lines.append(
        f"\nRecommendation: '{summary.iloc[0]['action']}' has the best average efficiency."
    )
    return "\n".join(lines)


@analyze_agent.tool
def compute_cost_efficiency(ctx: RunContext[DataDeps], entity_id: str) -> str:
    """Compute effect/cost ratio for every (action, scenario) combo for an entity."""
    df = ctx.deps.df
    rows = df[df["entity"] == entity_id].copy()
    if rows.empty:
        return f"No data found for entity '{entity_id}'."
    rows["efficiency"] = rows["intervention_effect"] / rows["intervention_cost"]
    rows = rows.sort_values("efficiency", ascending=False)
    return rows[
        ["action", "scenario", "intervention_effect", "intervention_cost", "efficiency"]
    ].to_string(index=False)


@analyze_agent.tool
def compare_entities(ctx: RunContext[DataDeps], entity_ids: list[str]) -> str:
    """Compare mean effect and mean cost across a list of entities."""
    df = ctx.deps.df
    rows = df[df["entity"].isin(entity_ids)]
    if rows.empty:
        return "No data found for the given entities."
    summary = (
        rows.groupby("entity")
        .agg(
            mean_effect=("intervention_effect", "mean"),
            mean_cost=("intervention_cost", "mean"),
            best_effect=("intervention_effect", "max"),
        )
        .reset_index()
        .sort_values("mean_effect", ascending=False)
    )
    return summary.to_string(index=False)


@analyze_agent.tool
def get_dataset_overview(ctx: RunContext[DataDeps]) -> str:
    """
    Return a compact aggregated overview of the entire dataset in a single call.
    Use this for global or all-entities queries instead of calling query_entity_data
    once per entity. Returns: global stats, mean effect by action, mean effect by
    scenario, and the top-5 / bottom-5 entities by mean effect.
    """
    df = ctx.deps.df
    global_stats = df[["intervention_effect", "intervention_cost"]].describe().round(2)
    by_action = (
        df.groupby("action")[["intervention_effect", "intervention_cost"]]
        .mean()
        .round(2)
    )
    by_scenario = (
        df.groupby("scenario")[["intervention_effect", "intervention_cost"]]
        .mean()
        .round(2)
    )
    by_entity = (
        df.groupby("entity")["intervention_effect"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"intervention_effect": "mean_effect"})
        .sort_values("mean_effect", ascending=False)
    )
    top5 = by_entity.head(5)
    bottom5 = by_entity.tail(5)
    return (
        f"=== Global Stats ===\n{global_stats.to_string()}\n\n"
        f"=== Mean by Action ===\n{by_action.to_string()}\n\n"
        f"=== Mean by Scenario ===\n{by_scenario.to_string()}\n\n"
        f"=== Top 5 Entities (mean effect) ===\n{top5.to_string(index=False)}\n\n"
        f"=== Bottom 5 Entities (mean effect) ===\n{bottom5.to_string(index=False)}"
    )


def get_analyze_agent() -> tuple[Agent, DataDeps]:
    return analyze_agent, DataDeps(df=get_df())
