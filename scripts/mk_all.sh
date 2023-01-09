#!/bin/sh

# makes specific experiment runs based on base_expr.sh 

for config_path in $@
do
    # take the full path 
    # reverse the path so that config_file is first
    # split on "/" and take the first element (the reversed config_file)
    # reverse the string to get the normal config_file
    # remove the .yml extension
    config_name=$(echo $config_path | rev | cut -d "/" -f 1 | rev | sed 's/\.yml//')
    expr_name=$(echo $config_name | cut -d "." -f 3)    # based on naming convention
    echo $expr_name
    
    script_path="run/all_$expr_name.sh"
    cp "scripts/base_all.sh" "$script_path"
    
    sed -i "s|EXPR_NAME|$expr_name|" "$script_path"
done

