#!/bin/bash
set -e
PYTHONPATH=. python3 exp_e1_slack_cdf.py
PYTHONPATH=. python3 exp_e2_mechanism.py
PYTHONPATH=. python3 exp_e3_sparsity_sweep.py
PYTHONPATH=. python3 exp_e4_real_trace.py
PYTHONPATH=. python3 exp_e5_burst_recovery.py
PYTHONPATH=. python3 exp_a1_bandwidth.py 
PYTHONPATH=. python3 exp_a2_distributions.py
echo "Done."