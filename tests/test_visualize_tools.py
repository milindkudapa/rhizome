"""Tests for Visualize Agent tool functions (no LLM calls)."""

from pathlib import Path

import pytest

from agents.visualize import PLOTS_DIR, visualize_agent
from tests.conftest import get_tool_fn


@pytest.fixture(scope="module")
def plot_bar():
    return get_tool_fn(visualize_agent, "plot_entity_bar_chart")


@pytest.fixture(scope="module")
def plot_scatter():
    return get_tool_fn(visualize_agent, "plot_cost_vs_effect_scatter")


@pytest.fixture(scope="module")
def plot_heatmap():
    return get_tool_fn(visualize_agent, "plot_global_heatmap")


# ── plot_entity_bar_chart ─────────────────────────────────────────────────────

class TestPlotEntityBarChart:
    def test_returns_path_string(self, ctx, plot_bar):
        result = plot_bar(ctx, "entity_1")
        assert isinstance(result, str)

    def test_saved_file_exists(self, ctx, plot_bar):
        path_str = plot_bar(ctx, "entity_1")
        assert Path(path_str).exists()

    def test_saved_file_is_png(self, ctx, plot_bar):
        path_str = plot_bar(ctx, "entity_1")
        assert path_str.endswith(".png")

    def test_filename_contains_entity_id(self, ctx, plot_bar):
        path_str = plot_bar(ctx, "entity_2")
        assert "entity_2" in path_str

    def test_file_is_non_empty(self, ctx, plot_bar):
        path_str = plot_bar(ctx, "entity_1")
        assert Path(path_str).stat().st_size > 0

    def test_unknown_entity_returns_message(self, ctx, plot_bar):
        result = plot_bar(ctx, "entity_999")
        assert "No data found" in result


# ── plot_cost_vs_effect_scatter ───────────────────────────────────────────────

class TestPlotCostVsEffectScatter:
    def test_returns_path_string(self, ctx, plot_scatter):
        result = plot_scatter(ctx, "entity_1")
        assert isinstance(result, str)

    def test_saved_file_exists(self, ctx, plot_scatter):
        path_str = plot_scatter(ctx, "entity_1")
        assert Path(path_str).exists()

    def test_saved_file_is_png(self, ctx, plot_scatter):
        path_str = plot_scatter(ctx, "entity_1")
        assert path_str.endswith(".png")

    def test_filename_contains_entity_id(self, ctx, plot_scatter):
        path_str = plot_scatter(ctx, "entity_2")
        assert "entity_2" in path_str

    def test_file_is_non_empty(self, ctx, plot_scatter):
        path_str = plot_scatter(ctx, "entity_1")
        assert Path(path_str).stat().st_size > 0

    def test_unknown_entity_returns_message(self, ctx, plot_scatter):
        result = plot_scatter(ctx, "entity_999")
        assert "No data found" in result


# ── plot_global_heatmap ───────────────────────────────────────────────────────

class TestPlotGlobalHeatmap:
    def test_returns_path_string(self, ctx, plot_heatmap):
        result = plot_heatmap(ctx)
        assert isinstance(result, str)

    def test_saved_file_exists(self, ctx, plot_heatmap):
        path_str = plot_heatmap(ctx)
        assert Path(path_str).exists()

    def test_saved_file_is_png(self, ctx, plot_heatmap):
        path_str = plot_heatmap(ctx)
        assert path_str.endswith(".png")

    def test_filename_is_global_heatmap(self, ctx, plot_heatmap):
        path_str = plot_heatmap(ctx)
        assert "global_heatmap" in Path(path_str).name

    def test_file_is_non_empty(self, ctx, plot_heatmap):
        path_str = plot_heatmap(ctx)
        assert Path(path_str).stat().st_size > 0


# ── plots/ directory ──────────────────────────────────────────────────────────

class TestPlotsDirectory:
    def test_plots_dir_exists(self):
        assert PLOTS_DIR.exists()
        assert PLOTS_DIR.is_dir()
