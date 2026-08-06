"""Tests for augur.report."""

from pathlib import Path

import pandas as pd

from augur.report import write_report


def test_write_report_creates_markdown_and_png(tmp_path: Path) -> None:
    """write_report produces a markdown file with the stats table and a non-empty sibling PNG."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
    report_path = tmp_path / "report.md"

    write_report(returns, report_path)

    content = report_path.read_text()
    for stat_name in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown"]:
        assert stat_name in content

    png_path = report_path.with_suffix(".png")
    assert png_path.exists()
    assert png_path.stat().st_size > 0
