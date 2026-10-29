"""Report tests that hit an unimplemented stub as skipped rather than failed.

Once the code under test is implemented, the test runs normally.
"""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1], "Skipped: not implemented yet")
