import pytest

failure_detected = False


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()
    global failure_detected
    if result.when == 'call' and result.outcome == 'failed':
        failure_detected = True


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if failure_detected:
        session.exitstatus = 1
    else:
        session.exitstatus = 0
