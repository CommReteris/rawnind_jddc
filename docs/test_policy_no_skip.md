# test_policy_no_skip.py Passing Tests Documentation

## test_no_skip_markers_in_acceptance_suite

This test enforces a zero-skip policy for acceptance tests by scanning all test files in the acceptance directory for pytest skip markers or calls. It checks for patterns like `pytest.mark.skip`, `@pytest.mark.skip`, and `pytest.skip()`. The test passes when no forbidden skip markers are found, ensuring that acceptance tests are not hiding regressions behind temporary skips and that all tests are actively running and validating the codebase.