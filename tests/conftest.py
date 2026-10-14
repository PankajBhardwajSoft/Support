import pytest

from ola_agent.netguard import block_network, unblock_network


@pytest.fixture(autouse=True)
def no_network():
    """Every test runs with outbound network blocked."""
    block_network()
    yield
    unblock_network()
