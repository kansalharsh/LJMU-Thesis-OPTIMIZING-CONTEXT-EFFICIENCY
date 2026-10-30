#!/usr/bin/env bash
# Recomputes thesis Tables 5.1 and 5.2 from the committed per-query records.
# No API key and no dataset needed.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.analyze --final
