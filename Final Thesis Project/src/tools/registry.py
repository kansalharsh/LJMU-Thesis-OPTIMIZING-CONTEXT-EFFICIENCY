"""Tool Access Layer: local stand-ins for the MCP-wrapped travel APIs.

Each function returns a JSON-serializable list of dicts, mirroring what a
real MCP tool call (Amadeus/Skyscanner-style) would hand back over
JSON-RPC. This is where Phase 2's Semantic Pruning Layer will intercept the
raw payload before it reaches the reasoning layer; the baseline agent
passes this output straight through, unpruned.
"""
from __future__ import annotations

import functools

import pandas as pd

from .. import config

MAX_ROWS = 50  # matches TravelPlanner's own tool-output truncation convention


@functools.lru_cache(maxsize=1)
def _flights_df() -> pd.DataFrame:
    return pd.read_csv(config.TOOLS_DB / "flights" / "clean_Flights_2022.csv")


@functools.lru_cache(maxsize=1)
def _accommodations_df() -> pd.DataFrame:
    return pd.read_csv(config.TOOLS_DB / "accommodations" / "clean_accommodations_2022.csv")


@functools.lru_cache(maxsize=1)
def _restaurants_df() -> pd.DataFrame:
    return pd.read_csv(config.TOOLS_DB / "restaurants" / "clean_restaurant_2022.csv")


@functools.lru_cache(maxsize=1)
def _attractions_df() -> pd.DataFrame:
    return pd.read_csv(config.TOOLS_DB / "attractions" / "attractions.csv")


@functools.lru_cache(maxsize=1)
def _distance_df() -> pd.DataFrame:
    return pd.read_csv(config.TOOLS_DB / "googleDistanceMatrix" / "distance.csv")


def _normalize_city(city: str) -> str:
    """Strips a trailing ', ST'/', State' suffix and normalizes case/whitespace.

    Models frequently call tools with "Washington, DC" rather than the bare
    "Washington" the CSVs key on; without this, exact-match lookups return 0
    rows and the agent (correctly) refuses to hallucinate a plan.
    """
    return city.split(",")[0].strip().lower()


def flight_search(origin: str, destination: str, date: str) -> list[dict]:
    df = _flights_df()
    hits = df[
        (df["OriginCityName"].str.lower() == _normalize_city(origin))
        & (df["DestCityName"].str.lower() == _normalize_city(destination))
        & (df["FlightDate"] == date)
    ]
    return hits.head(MAX_ROWS).to_dict(orient="records")


def accommodation_search(city: str) -> list[dict]:
    df = _accommodations_df()
    hits = df[df["city"].str.lower() == _normalize_city(city)]
    return hits.head(MAX_ROWS).to_dict(orient="records")


def restaurant_search(city: str) -> list[dict]:
    df = _restaurants_df()
    hits = df[df["City"].str.lower() == _normalize_city(city)]
    return hits.head(MAX_ROWS).to_dict(orient="records")


def attraction_search(city: str) -> list[dict]:
    df = _attractions_df()
    hits = df[df["City"].str.lower() == _normalize_city(city)]
    return hits.head(MAX_ROWS).to_dict(orient="records")


def distance_search(origin: str, destination: str) -> list[dict]:
    df = _distance_df()
    hits = df[
        (df["origin"].str.lower() == _normalize_city(origin))
        & (df["destination"].str.lower() == _normalize_city(destination))
    ]
    return hits.head(MAX_ROWS).to_dict(orient="records")


TOOLS = {
    "flight_search": {
        "fn": flight_search,
        "args": ["origin", "destination", "date"],
        "description": "Search flights between two cities on a given date (YYYY-MM-DD).",
    },
    "accommodation_search": {
        "fn": accommodation_search,
        "args": ["city"],
        "description": "Search accommodations available in a city.",
    },
    "restaurant_search": {
        "fn": restaurant_search,
        "args": ["city"],
        "description": "Search restaurants available in a city.",
    },
    "attraction_search": {
        "fn": attraction_search,
        "args": ["city"],
        "description": "Search attractions available in a city.",
    },
    "distance_search": {
        "fn": distance_search,
        "args": ["origin", "destination"],
        "description": "Get driving/transit distance and duration between two cities.",
    },
}


def call_tool(name: str, kwargs: dict) -> list[dict]:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOLS[name]["fn"](**kwargs)


def tool_manifest() -> str:
    lines = []
    for name, spec in TOOLS.items():
        lines.append(f"- {name}({', '.join(spec['args'])}): {spec['description']}")
    return "\n".join(lines)
