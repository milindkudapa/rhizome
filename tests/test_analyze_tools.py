"""Tests for Analyze Agent tool functions (no LLM calls)."""

import json

import pytest

from agents.analyze import analyze_agent
from tests.conftest import get_tool_fn


@pytest.fixture(scope="module")
def query_entity_data():
    return get_tool_fn(analyze_agent, "query_entity_data")


@pytest.fixture(scope="module")
def get_best_scenario():
    return get_tool_fn(analyze_agent, "get_best_scenario")


@pytest.fixture(scope="module")
def get_highest_effect():
    return get_tool_fn(analyze_agent, "get_highest_effect_scenario_across_actions")


@pytest.fixture(scope="module")
def recommend_action():
    return get_tool_fn(analyze_agent, "recommend_action")


@pytest.fixture(scope="module")
def compute_cost_efficiency():
    return get_tool_fn(analyze_agent, "compute_cost_efficiency")


@pytest.fixture(scope="module")
def compare_entities():
    return get_tool_fn(analyze_agent, "compare_entities")


# ── query_entity_data ─────────────────────────────────────────────────────────

class TestQueryEntityData:
    def test_returns_json_for_known_entity(self, ctx, query_entity_data):
        result = query_entity_data(ctx, "entity_1")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 9  # 3 actions × 3 scenarios

    def test_json_has_correct_columns(self, ctx, query_entity_data):
        data = json.loads(query_entity_data(ctx, "entity_1"))
        assert all(
            {"entity", "action", "scenario", "intervention_effect", "intervention_cost"}
            <= set(row.keys())
            for row in data
        )

    def test_only_returns_requested_entity(self, ctx, query_entity_data):
        data = json.loads(query_entity_data(ctx, "entity_2"))
        assert all(row["entity"] == "entity_2" for row in data)

    def test_missing_entity_returns_message(self, ctx, query_entity_data):
        result = query_entity_data(ctx, "entity_999")
        assert "No data found" in result


# ── get_best_scenario ─────────────────────────────────────────────────────────

class TestGetBestScenario:
    def test_returns_scenario_string(self, ctx, get_best_scenario):
        result = get_best_scenario(ctx, "entity_1", "action_1")
        # action_1 for entity_1: scenario_2 has effect=50 (highest)
        assert "scenario_2" in result
        assert "50.00" in result

    def test_correct_entity_and_action(self, ctx, get_best_scenario):
        result = get_best_scenario(ctx, "entity_1", "action_2")
        # action_2 for entity_1: scenario_2 has effect=40 (highest)
        assert "scenario_2" in result

    def test_unknown_entity_action(self, ctx, get_best_scenario):
        result = get_best_scenario(ctx, "entity_999", "action_1")
        assert "No data found" in result


# ── get_highest_effect_scenario_across_actions ────────────────────────────────

class TestGetHighestEffect:
    def test_finds_global_max_for_entity(self, ctx, get_highest_effect):
        result = get_highest_effect(ctx, "entity_1")
        # entity_1 max effect = 50.0 at action_1/scenario_2
        assert "50.00" in result
        assert "action_1" in result
        assert "scenario_2" in result

    def test_entity_2_max(self, ctx, get_highest_effect):
        result = get_highest_effect(ctx, "entity_2")
        # entity_2 max effect = 60.0 at action_1/scenario_1
        assert "60.00" in result

    def test_unknown_entity(self, ctx, get_highest_effect):
        assert "No data found" in get_highest_effect(ctx, "entity_999")


# ── recommend_action ──────────────────────────────────────────────────────────

class TestRecommendAction:
    def test_returns_string_with_recommendation(self, ctx, recommend_action):
        result = recommend_action(ctx, "entity_1")
        assert "Recommendation" in result
        assert "action_" in result

    def test_lists_all_actions(self, ctx, recommend_action):
        result = recommend_action(ctx, "entity_1")
        assert "action_1" in result
        assert "action_2" in result
        assert "action_3" in result

    def test_entity_1_recommends_action_2(self, ctx, recommend_action):
        # entity_1 action_2 efficiencies: 20/400=0.05, 40/50=0.80, 5/500=0.01 → mean=0.287
        # entity_1 action_1 efficiencies: 10/100=0.10, 50/200=0.25, 30/150=0.20 → mean=0.183
        # entity_1 action_3 efficiencies: 15/300=0.05, 25/250=0.10, 35/350=0.10 → mean=0.083
        result = recommend_action(ctx, "entity_1")
        assert "action_2" in result  # highest mean efficiency

    def test_unknown_entity(self, ctx, recommend_action):
        assert "No data found" in recommend_action(ctx, "entity_999")


# ── compute_cost_efficiency ───────────────────────────────────────────────────

class TestComputeCostEfficiency:
    def test_returns_table_with_efficiency_column(self, ctx, compute_cost_efficiency):
        result = compute_cost_efficiency(ctx, "entity_1")
        assert "efficiency" in result
        assert "action" in result
        assert "scenario" in result

    def test_nine_rows_for_entity(self, ctx, compute_cost_efficiency):
        result = compute_cost_efficiency(ctx, "entity_1")
        # 9 data rows + 1 header = 10 lines (plus possible blank lines)
        data_lines = [l for l in result.strip().splitlines() if l.strip()]
        assert len(data_lines) == 10  # 1 header + 9 data rows

    def test_unknown_entity(self, ctx, compute_cost_efficiency):
        assert "No data found" in compute_cost_efficiency(ctx, "entity_999")


# ── compare_entities ──────────────────────────────────────────────────────────

class TestCompareEntities:
    def test_both_entities_in_result(self, ctx, compare_entities):
        result = compare_entities(ctx, ["entity_1", "entity_2"])
        assert "entity_1" in result
        assert "entity_2" in result

    def test_sorted_by_mean_effect_descending(self, ctx, compare_entities):
        result = compare_entities(ctx, ["entity_1", "entity_2"])
        lines = [l for l in result.strip().splitlines() if l.strip()]
        # entity_2 has higher mean effect → should appear before entity_1
        e2_pos = next(i for i, l in enumerate(lines) if "entity_2" in l)
        e1_pos = next(i for i, l in enumerate(lines) if "entity_1" in l)
        assert e2_pos < e1_pos

    def test_single_entity(self, ctx, compare_entities):
        result = compare_entities(ctx, ["entity_1"])
        assert "entity_1" in result

    def test_unknown_entities(self, ctx, compare_entities):
        result = compare_entities(ctx, ["entity_999"])
        assert "No data found" in result
