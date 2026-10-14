"""Shared start-up for scripts: project on sys.path, config loaded, network blocked."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ola_agent import config  # noqa: E402,F401  (telemetry off before anything else)
from ola_agent.netguard import block_network  # noqa: E402

block_network()
