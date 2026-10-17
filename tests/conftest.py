"""Pytest configuration and environment fixtures for MedMLOps-X.

Ensures hermetic test execution with isolated credentials and temporary state.
"""

import pytest

TEST_ADMIN_API_KEY = "medmlops-admin-key-sprint6"


@pytest.fixture(autouse=True)
def configure_test_admin_key(monkeypatch: pytest.MonkeyPatch):
    """Ensure an explicit test administrative API key is set for authorized test requests.

    Individual security tests can unset, empty, or modify this using monkeypatch.
    """
    monkeypatch.setenv("MEDMLOPS_ADMIN_API_KEY", TEST_ADMIN_API_KEY)
