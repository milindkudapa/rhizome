"""Tests for Report Agent tool functions (no LLM calls)."""

from pathlib import Path

import pytest

from agents.report import REPORTS_DIR, report_agent
from tests.conftest import get_tool_fn


@pytest.fixture(scope="module")
def get_summary_stats():
    return get_tool_fn(report_agent, "get_summary_stats")


@pytest.fixture(scope="module")
def get_top_performers():
    return get_tool_fn(report_agent, "get_top_performers")


@pytest.fixture(scope="module")
def get_bottom_performers():
    return get_tool_fn(report_agent, "get_bottom_performers")


@pytest.fixture(scope="module")
def get_best_action_per_entity():
    return get_tool_fn(report_agent, "get_best_action_per_entity")


@pytest.fixture(scope="module")
def write_report_to_file():
    return get_tool_fn(report_agent, "write_report_to_file")


# ── get_summary_stats ─────────────────────────────────────────────────────────

class TestGetSummaryStats:
    def test_returns_string(self, ctx, get_summary_stats):
        assert isinstance(get_summary_stats(ctx), str)

    def test_contains_global_stats_section(self, ctx, get_summary_stats):
        assert "Global Stats" in get_summary_stats(ctx)

    def test_contains_action_section(self, ctx, get_summary_stats):
        assert "Mean by Action" in get_summary_stats(ctx)

    def test_contains_scenario_section(self, ctx, get_summary_stats):
        assert "Mean by Scenario" in get_summary_stats(ctx)

    def test_contains_statistical_measures(self, ctx, get_summary_stats):
        result = get_summary_stats(ctx)
        for stat in ("mean", "std", "min", "max"):
            assert stat in result

    def test_contains_all_actions(self, ctx, get_summary_stats):
        result = get_summary_stats(ctx)
        assert "action_1" in result
        assert "action_2" in result
        assert "action_3" in result


# ── get_top_performers ────────────────────────────────────────────────────────

class TestGetTopPerformers:
    def test_default_returns_string(self, ctx, get_top_performers):
        assert isinstance(get_top_performers(ctx), str)

    def test_respects_n_limit(self, ctx, get_top_performers):
        result = get_top_performers(ctx, n=1)
        # Only 2 entities in sample_df — n=1 should show just 1
        data_lines = [l for l in result.strip().splitlines() if l.strip()]
        # 1 header + 1 data row
        assert len(data_lines) == 2

    def test_contains_mean_effect_column(self, ctx, get_top_performers):
        assert "mean_effect" in get_top_performers(ctx)

    def test_entity_2_ranks_first(self, ctx, get_top_performers):
        result = get_top_performers(ctx, n=2)
        lines = [l for l in result.strip().splitlines() if l.strip()]
        # First data line (after header) should be entity with higher mean effect
        # entity_2 mean effect > entity_1 mean effect
        assert "entity_2" in lines[1]


# ── get_bottom_performers ─────────────────────────────────────────────────────

class TestGetBottomPerformers:
    def test_returns_string(self, ctx, get_bottom_performers):
        assert isinstance(get_bottom_performers(ctx), str)

    def test_respects_n_limit(self, ctx, get_bottom_performers):
        result = get_bottom_performers(ctx, n=1)
        data_lines = [l for l in result.strip().splitlines() if l.strip()]
        assert len(data_lines) == 2  # header + 1 row

    def test_entity_1_ranks_first(self, ctx, get_bottom_performers):
        # entity_1 has lower mean effect than entity_2
        result = get_bottom_performers(ctx, n=2)
        lines = [l for l in result.strip().splitlines() if l.strip()]
        assert "entity_1" in lines[1]


# ── get_best_action_per_entity ────────────────────────────────────────────────

class TestGetBestActionPerEntity:
    def test_returns_string(self, ctx, get_best_action_per_entity):
        assert isinstance(get_best_action_per_entity(ctx), str)

    def test_contains_frequency_section(self, ctx, get_best_action_per_entity):
        assert "frequency" in get_best_action_per_entity(ctx).lower()

    def test_contains_action_names(self, ctx, get_best_action_per_entity):
        result = get_best_action_per_entity(ctx)
        assert "action_" in result

    def test_entity_1_best_action_is_action_1(self, ctx, get_best_action_per_entity):
        # entity_1: action_1 mean=(10+50+30)/3=30, action_2=(20+40+5)/3=21.7, action_3=(15+25+35)/3=25
        # action_1 wins for entity_1
        result = get_best_action_per_entity(ctx)
        # Find the line containing entity_1 and check the action
        lines = result.splitlines()
        entity_1_line = next((l for l in lines if "entity_1" in l), "")
        assert "action_1" in entity_1_line


# ── write_report_to_file ──────────────────────────────────────────────────────

class TestWriteReportToFile:
    CONTENT = "# Test Report\n\nThis is a test."

    def test_returns_path_string(self, ctx, write_report_to_file):
        result = write_report_to_file(ctx, self.CONTENT)
        assert isinstance(result, str)

    def test_saved_file_exists(self, ctx, write_report_to_file):
        path_str = write_report_to_file(ctx, self.CONTENT)
        assert Path(path_str).exists()

    def test_saved_file_is_markdown(self, ctx, write_report_to_file):
        path_str = write_report_to_file(ctx, self.CONTENT)
        assert path_str.endswith(".md")

    def test_saved_file_contains_content(self, ctx, write_report_to_file):
        path_str = write_report_to_file(ctx, self.CONTENT)
        assert Path(path_str).read_text(encoding="utf-8") == self.CONTENT

    def test_each_call_creates_unique_file(self, ctx, write_report_to_file):
        import time
        path1 = write_report_to_file(ctx, "report 1")
        time.sleep(1.1)  # timestamp granularity is 1s
        path2 = write_report_to_file(ctx, "report 2")
        assert path1 != path2


# ── reports/ directory ────────────────────────────────────────────────────────

class TestReportsDirectory:
    def test_reports_dir_exists(self):
        assert REPORTS_DIR.exists()
        assert REPORTS_DIR.is_dir()
