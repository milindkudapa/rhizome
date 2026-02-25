"""Shared pytest fixtures."""

# Load .env BEFORE any agents import so PydanticAI can resolve ANTHROPIC_API_KEY.
from dotenv import load_dotenv

load_dotenv()

from types import SimpleNamespace  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from agents.models import DataDeps  # noqa: E402


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Minimal in-memory DataFrame mirroring synthetic_data.csv structure."""
    rows = [
        # entity_1
        ("entity_1", "action_1", "scenario_1", 10.0, 100.0),
        ("entity_1", "action_1", "scenario_2", 50.0, 200.0),
        ("entity_1", "action_1", "scenario_3", 30.0, 150.0),
        ("entity_1", "action_2", "scenario_1", 20.0, 400.0),
        ("entity_1", "action_2", "scenario_2", 40.0, 50.0),
        ("entity_1", "action_2", "scenario_3",  5.0, 500.0),
        ("entity_1", "action_3", "scenario_1", 15.0, 300.0),
        ("entity_1", "action_3", "scenario_2", 25.0, 250.0),
        ("entity_1", "action_3", "scenario_3", 35.0, 350.0),
        # entity_2
        ("entity_2", "action_1", "scenario_1", 60.0, 120.0),
        ("entity_2", "action_1", "scenario_2",  8.0,  80.0),
        ("entity_2", "action_1", "scenario_3", 45.0,  90.0),
        ("entity_2", "action_2", "scenario_1", 55.0, 110.0),
        ("entity_2", "action_2", "scenario_2", 12.0, 200.0),
        ("entity_2", "action_2", "scenario_3", 22.0, 220.0),
        ("entity_2", "action_3", "scenario_1", 33.0, 330.0),
        ("entity_2", "action_3", "scenario_2", 44.0, 440.0),
        ("entity_2", "action_3", "scenario_3", 11.0, 110.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["entity", "action", "scenario", "intervention_effect", "intervention_cost"],
    )


@pytest.fixture(scope="session")
def deps(sample_df) -> DataDeps:
    return DataDeps(df=sample_df)


@pytest.fixture(scope="session")
def ctx(deps) -> SimpleNamespace:
    """Minimal mock RunContext — only .deps is needed by tool functions."""
    return SimpleNamespace(deps=deps)


def get_tool_fn(agent, name):
    """Return the raw callable for a named tool registered on an agent."""
    return agent._function_toolset.tools[name].function
