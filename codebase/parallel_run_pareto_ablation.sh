#!/bin/bash

for p_idx in {1..10}
do
    screen -S "par_causal_pareto_ablation$p_idx" -d -m bash -c 'source activate_env.sh; python cfr_param_search.py configs/pareto_ablation.txt 30; sleep 0.5s; '
    echo "************************************************"
done

