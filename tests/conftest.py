import pytest

from ola_agent.cache import CACHE
from ola_agent.netguard import block_network, unblock_network


@pytest.fixture(autouse=True)
def no_network():
    """Every test runs with outbound network blocked."""
    block_network()
    yield
    unblock_network()


@pytest.fixture(autouse=True)
def empty_cache():
    """Each test starts with an empty response cache, so one test's answers never reach another."""
    CACHE.clear()
    yield
    CACHE.clear()
