"""Experimental condition: Semantic-Aware MCP Gateway (Recursive Semantic Pruning).

Prunes each raw tool Observation down to the most relevant rows before it
reaches the reasoning layer (see semantic_gateway.py for the embedding/
scoring/pruning steps), and exposes a `fetch_more` pseudo-tool so the
agent can pull back a specific prior call's pruned rows if it decides the
pruned set wasn't enough — the "feedback loop" from the proposal.
"""
from __future__ import annotations

import json

from .. import semantic_gateway
from ..data_loader import TravelQuery
from .base_agent import BaseReActAgent, AgentRunResult
from .. import config

FETCH_MORE_MANIFEST = (
    '\n- fetch_more(tool_name): Retrieve the next batch of lower-relevance results that '
    "were pruned from a previous call to the given tool_name."
)


class GatewayAgent(BaseReActAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self._pruned_cache: dict[str, list[dict]] = {}
        self._intent: str = ""
        self.total_rows_seen = 0
        self.total_rows_pruned = 0

    def extra_tool_manifest(self) -> str:
        return FETCH_MORE_MANIFEST

    def run(self, query: TravelQuery, max_steps: int = config.MAX_REACT_STEPS) -> AgentRunResult:
        self._pruned_cache = {}
        self._intent = (
            f"{query.query} Destination: {query.dest}. Budget: ${query.budget}. "
            f"Constraints: {query.local_constraint}"
        )
        self.total_rows_seen = 0
        self.total_rows_pruned = 0
        return super().run(query, max_steps)

    def call_tool(self, tool_name: str, kwargs: dict) -> list[dict]:
        if tool_name == "fetch_more":
            target = kwargs.get("tool_name", "")
            pruned = self._pruned_cache.get(target, [])
            batch, rest = pruned[:10], pruned[10:]
            self._pruned_cache[target] = rest
            return batch
        return super().call_tool(tool_name, kwargs)

    def transform_observation(self, tool_name: str, observation: list[dict]) -> str:
        if tool_name == "fetch_more":
            return json.dumps(observation)

        self.total_rows_seen += len(observation)
        kept, pruned = semantic_gateway.constraint_aware_prune(
            observation, self._intent, price_floor_keep=config.PRICE_FLOOR_KEEP
        )
        self.total_rows_pruned += len(pruned)

        if pruned:
            self._pruned_cache[tool_name] = pruned
            note = (
                f" [{len(pruned)} lower-relevance result(s) pruned for context efficiency. "
                f'If these aren\'t enough, call Action: fetch_more(tool_name="{tool_name}")]'
            )
        else:
            note = ""
        return json.dumps(kept) + note
