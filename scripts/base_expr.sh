#!/bin/bash

declare -i length; length=0                     # set integer attribute

for i in "${@}"; do length+=1; done             # count number of arguments

# When running with no arguments use default
# Seeded runs are used for batched train-test in .../run/all_<expr_name>

if [ $length = 0 ]; then
    python3 src/main.py -v --config CONFIG_PATH    # placeholders for specific experiment runs
else
    python3 -W ignore src/main.py --config CONFIG_PATH --seed $1
fi
