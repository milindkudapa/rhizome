"""Tests for agents/models.py — Pydantic model validation."""

import pandas as pd
import pytest
from pydantic import ValidationError

from agents.models import (
    AnalysisResult,
    DataDeps,
    OrchestratorResponse,
    ReportResult,
    VisualizationResult,
)


# ── DataDeps ──────────────────────────────────────────────────────────────────

class TestDataDeps:
    def test_accepts_dataframe(self, sample_df):
        deps = DataDeps(df=sample_df)
        assert isinstance(deps.df, pd.DataFrame)

    def test_rejects_non_dataframe(self):
        with pytest.raises(ValidationError):
            DataDeps(df="not a dataframe")  # type: ignore[arg-type]

    def test_missing_df_raises(self):
        with pytest.raises(ValidationError):
            DataDeps()  # type: ignore[call-arg]


# ── AnalysisResult ────────────────────────────────────────────────────────────

class TestAnalysisResult:
    def test_minimal_valid(self):
        r = AnalysisResult(answer="entity_1 best action is action_2")
        assert r.answer == "entity_1 best action is action_2"
        assert r.recommended_action is None
        assert r.key_findings == []

    def test_full_valid(self):
        r = AnalysisResult(
            answer="detailed answer",
            recommended_action="action_2",
            key_findings=["finding 1", "finding 2"],
        )
        assert r.recommended_action == "action_2"
        assert len(r.key_findings) == 2

    def test_missing_answer_raises(self):
        with pytest.raises(ValidationError):
            AnalysisResult()  # type: ignore[call-arg]

    def test_key_findings_defaults_to_empty_list(self):
        r = AnalysisResult(answer="ok")
        assert r.key_findings == []

    def test_serialises_to_json(self):
        r = AnalysisResult(answer="ok", recommended_action="action_1")
        data = r.model_dump()
        assert data["answer"] == "ok"
        assert data["recommended_action"] == "action_1"
        assert "key_findings" in data


# ── VisualizationResult ───────────────────────────────────────────────────────

class TestVisualizationResult:
    def test_valid(self):
        r = VisualizationResult(
            plot_paths=["/tmp/plot.png"],
            description="Bar chart of entity_1",
        )
        assert len(r.plot_paths) == 1
        assert "entity_1" in r.description

    def test_multiple_paths(self):
        r = VisualizationResult(
            plot_paths=["/tmp/a.png", "/tmp/b.png"],
            description="Two charts",
        )
        assert len(r.plot_paths) == 2

    def test_missing_fields_raise(self):
        with pytest.raises(ValidationError):
            VisualizationResult(plot_paths=["/tmp/a.png"])  # type: ignore[call-arg]

    def test_empty_plot_paths_allowed(self):
        r = VisualizationResult(plot_paths=[], description="No plots yet")
        assert r.plot_paths == []


# ── ReportResult ──────────────────────────────────────────────────────────────

class TestReportResult:
    def test_valid(self):
        r = ReportResult(
            file_path="/tmp/report.md",
            highlights=["finding A", "finding B"],
        )
        assert r.file_path == "/tmp/report.md"
        assert len(r.highlights) == 2

    def test_missing_file_path_raises(self):
        with pytest.raises(ValidationError):
            ReportResult(highlights=["x"])  # type: ignore[call-arg]

    def test_empty_highlights_allowed(self):
        r = ReportResult(file_path="/tmp/r.md", highlights=[])
        assert r.highlights == []

    def test_serialises(self):
        r = ReportResult(file_path="/tmp/r.md", highlights=["h1"])
        assert r.model_dump()["file_path"] == "/tmp/r.md"


# ── OrchestratorResponse ──────────────────────────────────────────────────────

class TestOrchestratorResponse:
    def test_minimal(self):
        r = OrchestratorResponse(answer="Here is the result.")
        assert r.plot_paths == []
        assert r.report_path is None

    def test_with_plots_and_report(self):
        r = OrchestratorResponse(
            answer="Done.",
            plot_paths=["/tmp/a.png"],
            report_path="/tmp/report.md",
        )
        assert len(r.plot_paths) == 1
        assert r.report_path == "/tmp/report.md"

    def test_missing_answer_raises(self):
        with pytest.raises(ValidationError):
            OrchestratorResponse()  # type: ignore[call-arg]
