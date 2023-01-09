#!/bin/bash

timestamp=$(date +"%m-%d-%T")
out_dir="output/EXPR_NAME"
out_path="$out_dir/$timestamp"   # output path based on time

if [[ ! -d $out_dir ]]; then     # create directory for experiment
    mkdir $out_dir
fi

touch $out_path                 # initialise empty output file
echo "Saving runs to $out_path"

# Run a set of 10 runs with the prechosen seeds
# Using grep and sed extract just the test accuracy

./run/EXPR_NAME.sh 510223 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" > $out_path &
./run/EXPR_NAME.sh 873616 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 557633 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 633554 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 493164 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 374916 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 80712  | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 863742 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 374967 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
./run/EXPR_NAME.sh 179277 | grep "test_acc" | sed -r "s/[ ]*test_acc[ ]*([0-9.]*)/\1/" >> $out_path &
