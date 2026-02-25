"""Tests for data/loader.py."""

import pandas as pd

from data.loader import DATA_PATH, get_df


def test_data_file_exists():
    assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"


def test_returns_dataframe():
    df = get_df()
    assert isinstance(df, pd.DataFrame)


def test_expected_columns():
    df = get_df()
    assert list(df.columns) == [
        "entity", "action", "scenario", "intervention_effect", "intervention_cost"
    ]


def test_row_count():
    df = get_df()
    # 99 entities × 3 actions × 3 scenarios
    assert len(df) == 891


def test_no_nulls():
    df = get_df()
    assert df.isnull().sum().sum() == 0


def test_entity_count():
    df = get_df()
    assert df["entity"].nunique() == 99


def test_action_values():
    df = get_df()
    assert set(df["action"].unique()) == {"action_1", "action_2", "action_3"}


def test_scenario_values():
    df = get_df()
    assert set(df["scenario"].unique()) == {"scenario_1", "scenario_2", "scenario_3"}


def test_effect_and_cost_are_positive():
    df = get_df()
    assert (df["intervention_effect"] > 0).all()
    assert (df["intervention_cost"] > 0).all()


def test_singleton_returns_same_object():
    df1 = get_df()
    df2 = get_df()
    assert df1 is df2
