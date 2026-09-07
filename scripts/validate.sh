#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || (${1:-cpu} != cpu && ${1:-cpu} != cuda) ]]; then
    echo "Usage: $0 [cpu|cuda]" >&2
    exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"
uv sync --locked
uv run --frozen python -m hedging_gym.validate --device "${1:-cpu}"
uv run --frozen python -m pytest -p no:cacheprovider -q \
    tests/test_finance.py tests/test_execution.py tests/test_gym.py \
    tests/test_benchmark.py tests/test_markets.py
