"""Tests for augur.report."""

from pathlib import Path

import numpy as np
import pandas as pd

from augur.report import write_report


def test_write_report_creates_markdown_and_png(tmp_path: Path) -> None:
    """write_report produces a markdown file with the stats table and a non-empty sibling PNG."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [-0.5] * 5}, index=dates)
    report_path = tmp_path / "report.md"

    write_report(returns, weights, report_path)

    content = report_path.read_text()
    for stat_name in [
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "avg_turnover_per_rebalance",
        "avg_gross_exposure",
        "avg_net_exposure",
    ]:
        assert stat_name in content

    png_path = report_path.with_suffix(".png")
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_write_report_includes_deflation_stats_when_trial_sharpes_given(
    tmp_path: Path,
) -> None:
    """Passing trial_sharpe_ratios should add n_trials/DSR rows to the table."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [-0.5] * 5}, index=dates)
    trial_sharpe_ratios = np.array([-0.1, 0.0, 0.1, 0.2])
    report_path = tmp_path / "report.md"

    write_report(returns, weights, report_path, trial_sharpe_ratios)

    content = report_path.read_text()
    for stat_name in ["n_trials", "expected_max_sharpe_ratio", "deflated_sharpe_ratio"]:
        assert stat_name in content


def test_write_report_omits_deflation_stats_by_default(tmp_path: Path) -> None:
    """Without trial_sharpe_ratios, the deflation rows should not appear."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [-0.5] * 5}, index=dates)
    report_path = tmp_path / "report.md"

    write_report(returns, weights, report_path)

    content = report_path.read_text()
    assert "deflated_sharpe_ratio" not in content


def test_write_report_includes_regression_stats_when_benchmark_given(tmp_path: Path) -> None:
    """Passing benchmark_returns should add beta/alpha_annualized rows to the table."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [-0.5] * 5}, index=dates)
    benchmark_returns = pd.Series([0.005, -0.01, 0.02, 0.001, 0.008], index=dates)
    report_path = tmp_path / "report.md"

    write_report(returns, weights, report_path, benchmark_returns=benchmark_returns)

    content = report_path.read_text()
    for stat_name in ["beta", "alpha_annualized"]:
        assert stat_name in content


def test_write_report_omits_regression_stats_by_default(tmp_path: Path) -> None:
    """Without benchmark_returns, the beta/alpha rows should not appear."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [-0.5] * 5}, index=dates)
    report_path = tmp_path / "report.md"

    write_report(returns, weights, report_path)

    content = report_path.read_text()
    assert "beta" not in content
