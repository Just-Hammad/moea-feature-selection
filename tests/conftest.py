import warnings

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: empirical checks that take tens of seconds")
    # scipy emits these for degenerate Wilcoxon inputs (two selectors identical
    # in every block), which is a real and reportable situation, not an error.
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")
