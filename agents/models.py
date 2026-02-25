"""Shared Pydantic output models for all agents."""

from pydantic import BaseModel, ConfigDict, Field

import pandas as pd


# ── Deps (shared state injected into each agent) ──────────────────────────────

class DataDeps(BaseModel):
    """Shared deps carrying the loaded DataFrame into any agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame


# ── Structured output types ────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """Structured output from the Analyze Agent."""

    answer: str = Field(
        description="Complete answer to the analysis query, citing specific numbers."
    )
    recommended_action: str | None = Field(
        None,
        description="Best action for the entity if a recommendation was requested.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Up to five bullet-point findings extracted from the data.",
    )


class VisualizationResult(BaseModel):
    """Structured output from the Visualize Agent."""

    plot_paths: list[str] = Field(
        description="Absolute file paths of every PNG saved during this request."
    )
    description: str = Field(
        description="What the chart(s) show and any notable visual patterns."
    )


class ReportResult(BaseModel):
    """Structured output from the Report Agent."""

    file_path: str = Field(
        description="Absolute path to the saved markdown report file."
    )
    highlights: list[str] = Field(
        description="Top 3-5 bullet-point highlights from the report."
    )


class OrchestratorResponse(BaseModel):
    """Structured output from the Orchestrator — what is shown to the user."""

    answer: str = Field(
        description="Full response to relay to the user, incorporating all agent findings."
    )
    plot_paths: list[str] = Field(
        default_factory=list,
        description="Paths to any PNG plots generated during this turn.",
    )
    report_path: str | None = Field(
        None,
        description="Path to any markdown report generated during this turn.",
    )
