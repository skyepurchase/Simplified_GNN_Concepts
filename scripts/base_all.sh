#!/bin/bash

timestamp=$(date +"%m-%d-%T")
out_dir="output/EXPR_NAME"
out_path="$out_dir/$timestamp"   # output path based on time

if [[ ! -d $out_dir ]]; then     # create directory for experiment
    mkdir $out_dir
fi

touch $out_path                 # initialise empty output file
echo "Saving runs to $out_path"

# array of random seeds
declare -a seeds=(510223 873616 557633 633554 493164 374916 80712 863742 374967 179277)

# Run a set of 10 runs with the prechosen seeds
# Using grep and sed extract just the test accuracy

for seed in "${seeds[@]}"; do
    ./run/EXPR_NAME.sh $seed                        \
        | grep "test_acc"                           \
        | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/"  \
        >> $out_path
done
