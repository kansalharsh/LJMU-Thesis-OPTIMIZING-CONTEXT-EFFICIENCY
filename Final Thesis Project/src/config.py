"""Central configuration: paths and environment-driven settings.

Everything that varies between machines or experiment conditions is read from
environment variables, so a run can be reproduced by exporting the same values.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# TravelPlanner data (not redistributed here - see data/README.md).
# Expected layout:  <DATASET_ROOT>/{Query database,Tools database}/...
DATASET_ROOT = Path(os.environ.get("THESIS_DATASET_ROOT", REPO_ROOT / "data"))
QUERY_DB = DATASET_ROOT / "Query database"
TOOLS_DB = DATASET_ROOT / "Tools database"

RESULTS_DIR = REPO_ROOT / "results"

# Reasoning-layer LLM. The thesis results use provider=openai, model=gpt-5.2.
LLM_PROVIDER = os.environ.get("THESIS_LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Gateway: number of lowest-priced rows always kept regardless of semantic score.
# Thesis batches 1-3 were run with 3; batch 4 (and the repo default) uses 6.
PRICE_FLOOR_KEEP = int(os.environ.get("THESIS_PRICE_FLOOR", "6"))

# Sampling temperature sent to the OpenAI API. "default" omits the parameter (provider default),
# which is how thesis batches 1-3 were run; batch 4 and the repo default use 0.
TEMPERATURE = os.environ.get("THESIS_TEMPERATURE", "0")

# Maximum Reason-Act steps per query before the run is recorded as max_steps_exceeded.
MAX_REACT_STEPS = int(os.environ.get("THESIS_MAX_STEPS", "12"))
