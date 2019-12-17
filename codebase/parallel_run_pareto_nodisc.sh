#!/bin/bash

for p_idx in {1..10}
do
    screen -S "par_causal_pareto_no_disc$p_idx" -d -m bash -c 'source activate_env.sh; python cfr_param_search.py configs/test_pareto_nodisc.txt 100000; sleep 0.5s; '
    echo "************************************************"
done

