"""Pytest configuration and session-level isolation fixtures.

Ensures that test suite execution does not mutate tracked repository artifacts
such as Results/nlp_drift_report.json and monitoring/nlp_retraining_history.json.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def preserve_tracked_reporting_artifacts():
    """Preserve tracked audit and monitoring artifacts across the test session."""
    drift_report_path = Path("Results/nlp_drift_report.json")
    history_path = Path("monitoring/nlp_retraining_history.json")

    drift_report_orig = (
        drift_report_path.read_bytes() if drift_report_path.exists() else None
    )
    history_orig = history_path.read_bytes() if history_path.exists() else None

    yield

    # Restore original content after all tests have completed
    if drift_report_orig is not None:
        drift_report_path.write_bytes(drift_report_orig)
    if history_orig is not None:
        history_path.write_bytes(history_orig)
