"""
Integration tests — one real LLM call per agent to verify the pipeline works end-to-end.

Run with:
    uv run pytest tests/test_integration.py -v

Skip with:
    uv run pytest -m "not integration"
"""

from pathlib import Path

import pytest

from agents.analyze import analyze_agent
from agents.models import (
    AnalysisResult,
    DataDeps,
    OrchestratorResponse,
    ReportResult,
    VisualizationResult,
)
from agents.orchestrator import orchestrator
from agents.report import report_agent
from agents.visualize import visualize_agent
from data.loader import get_df

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_deps() -> DataDeps:
    return DataDeps(df=get_df())


def test_analyze_agent(real_deps):
    """Single query — verifies structured output type and non-empty answer."""
    result = analyze_agent.run_sync(
        "For entity_1, which action has the highest mean effect?",
        deps=real_deps,
    )
    out = result.output
    assert isinstance(out, AnalysisResult)
    assert isinstance(out.answer, str) and out.answer
    assert isinstance(out.key_findings, list)


def test_visualize_agent(real_deps):
    """Single plot request — verifies PNG is saved and paths returned."""
    result = visualize_agent.run_sync(
        "Create a bar chart for entity_1.",
        deps=real_deps,
    )
    out = result.output
    assert isinstance(out, VisualizationResult)
    assert len(out.plot_paths) > 0
    assert all(Path(p).exists() for p in out.plot_paths)
    assert isinstance(out.description, str) and out.description


def test_report_agent(real_deps):
    """Single report request — verifies markdown file is saved."""
    result = report_agent.run_sync(
        "Write a brief one-paragraph summary report of the dataset.",
        deps=real_deps,
    )
    out = result.output
    assert isinstance(out, ReportResult)
    assert out.file_path.endswith(".md")
    assert Path(out.file_path).exists()
    assert len(out.highlights) > 0


def test_orchestrator(real_deps):
    """Single analysis query through the orchestrator — verifies routing and response."""
    result = orchestrator.run_sync(
        "What action should we consider for entity_1?",
        deps=real_deps,
    )
    out = result.output
    assert isinstance(out, OrchestratorResponse)
    assert isinstance(out.answer, str) and out.answer
    assert isinstance(out.plot_paths, list)
