#!/bin/sh

# Uses the src/optimise.py script to search for the optimal hyperparameters for given configs

for config_path in $@; do
    echo $config_path
    python3 src/optimise.py --config $config_path --root /data --min_decay 0.01 --max_decay 1.0 --min_lr 0.000001 --max_lr 0.01 --epochs 5000
done

