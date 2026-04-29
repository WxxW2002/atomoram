#!/bin/bash
set -e
PYTHONPATH=. python3 scripts/Fig3_mechanism_validation.py
PYTHONPATH=. python3 scripts/Fig4_sparsity_sweep.py
PYTHONPATH=. python3 scripts/Fig5_burst_recovery.py
PYTHONPATH=. python3 scripts/Fig6_slack_cdf.py
PYTHONPATH=. python3 scripts/Fig7_Fig8_distributions.py
PYTHONPATH=. python3 scripts/Tab3_real_trace_latency.py
echo "Done."