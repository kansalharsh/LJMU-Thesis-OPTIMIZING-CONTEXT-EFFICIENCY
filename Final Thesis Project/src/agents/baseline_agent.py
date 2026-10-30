"""Baseline condition: standard MCP-style ReAct agent, no pruning gateway.

This is the control group the Semantic-Aware Gateway (Phase 2, not yet
built) will be compared against on TCR / CSR / token-to-signal ratio /
latency.
"""
from .base_agent import BaseReActAgent


class BaselineAgent(BaseReActAgent):
    pass
