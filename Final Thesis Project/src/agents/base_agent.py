"""Shared ReAct loop (Reasoning Layer) driving tool calls to a final plan.

Subclasses control one thing: how a raw tool observation is transformed
before being appended to context. The baseline agent passes it through
unchanged; the Phase 2 gateway agent will prune it here instead.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from .. import config
from ..data_loader import TravelQuery
from ..llm_provider import LLMResponse
from ..tools import registry

SYSTEM_PROMPT = """You are a travel-planning agent. You must produce a complete day-by-day \
itinerary (transportation, daily meals, attractions, accommodation) that satisfies the \
user's request and stated constraints.

Available tools:
{tool_manifest}

You MUST call flight_search, accommodation_search, restaurant_search, and attraction_search \
at least once each for the destination city before producing the Final Plan. Every meal, \
attraction, flight, and accommodation in the plan must come from a tool Observation you were \
given — never invent one from general knowledge, even if it means the plan is incomplete.

To call a tool, respond with EXACTLY one line:
Action: tool_name(arg1="value1", arg2="value2")

After a tool call you will be given an Observation. You may call more tools. \
When you have called all four required tool types at least once, respond with:
Final Plan:
<the complete day-by-day itinerary, ending with a line "Total Cost: $<amount>">

Do not call more than one tool per turn. Do not explain your reasoning outside of the plan.
"""

ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\)", re.DOTALL)
ARG_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


@dataclass
class AgentRunResult:
    plan_text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    rate_limit_wait_s: float
    steps: int
    error: str | None = None


class BaseReActAgent:
    def __init__(self, llm):
        self.llm = llm

    def extra_tool_manifest(self) -> str:
        """Extra tool descriptions to append to the manifest. Override in subclasses."""
        return ""

    def call_tool(self, tool_name: str, kwargs: dict) -> list[dict]:
        """Dispatches a tool call. Override to intercept gateway-only pseudo-tools."""
        return registry.call_tool(tool_name, kwargs)

    def transform_observation(self, tool_name: str, observation: list[dict]) -> str:
        """Baseline: raw JSON, unpruned. Override in the gateway agent."""
        return json.dumps(observation)

    def run(self, query: TravelQuery, max_steps: int = config.MAX_REACT_STEPS) -> AgentRunResult:
        prompt = SYSTEM_PROMPT.format(
            tool_manifest=registry.tool_manifest() + self.extra_tool_manifest()
        )
        prompt += f"\n\nUser request: {query.query}\n"

        total_in = total_out = 0
        total_latency = total_wait = 0.0
        transcript = prompt

        for step in range(1, max_steps + 1):
            print(f"    step {step}/{max_steps}...", file=sys.stderr)
            try:
                resp: LLMResponse = self.llm.generate(transcript)
            except Exception as e:  # noqa: BLE001 - surfaced to caller via error field
                return AgentRunResult("", total_in, total_out, total_latency, total_wait, step, str(e))

            total_in += resp.input_tokens
            total_out += resp.output_tokens
            total_latency += resp.latency_s
            total_wait += resp.rate_limit_wait_s

            if "Final Plan:" in resp.text:
                plan_text = resp.text.split("Final Plan:", 1)[1].strip()
                return AgentRunResult(plan_text, total_in, total_out, total_latency, total_wait, step)

            match = ACTION_RE.search(resp.text)
            if not match:
                # Model didn't follow the protocol; treat remaining text as the plan.
                return AgentRunResult(resp.text.strip(), total_in, total_out, total_latency, total_wait, step)

            tool_name, arg_str = match.group(1), match.group(2)
            kwargs = dict(ARG_RE.findall(arg_str))
            try:
                observation = self.call_tool(tool_name, kwargs)
            except Exception as e:  # noqa: BLE001
                observation = [{"error": str(e)}]

            observation_text = self.transform_observation(tool_name, observation)
            transcript += f"\n{resp.text.strip()}\nObservation: {observation_text}\n"

        return AgentRunResult("", total_in, total_out, total_latency, total_wait, max_steps, "max_steps_exceeded")
