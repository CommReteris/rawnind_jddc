# test_policy_no_xfail.py Passing Tests Documentation

## test_no_xfail_markers_in_acceptance_suite

This test enforces a zero-xfail policy for acceptance tests by scanning all test files in the acceptance directory for pytest xfail markers or calls. It checks for patterns like `pytest.mark.xfail`, `@pytest.mark.xfail`, and `pytest.xfail()`. The test passes when no forbidden expected-fail markers are found, ensuring that acceptance tests are not hiding regressions behind expected-fail annotations and that all tests are expected to pass.