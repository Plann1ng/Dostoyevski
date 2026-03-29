#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U ccxt pandas numpy
python final_backtest_v2_exchange.py "$@"
