import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    # every test starts from the same RNG state, so CI runs never depend on draw order or luck
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or "slow" in (config.getoption("-m") or ""):
        return
    skip_slow = pytest.mark.skip(reason="slow test: run with --runslow or -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
