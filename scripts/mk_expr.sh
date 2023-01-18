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
    dataset=$(echo $config_name | cut -d "." -f 3)    # based on naming convention
    model=$(echo $config_name | cut -d "." -f 1)
    echo $model-$dataset
    
    script_path="run/$model-$dataset.sh"
    cp "scripts/base_expr.sh" "$script_path"
    
    # check whether path was absolute or relative to home
    config_path_start=$(echo $config_path | cut -c1-1)
    if [ "$config_path_start" == "/" ] || [ "$config_path_start" == "~" ]; then
        sed -i "s|CONFIG_PATH|$config_path|" "$script_path"
    else
        sed -i "s|CONFIG_PATH|$(pwd)/$config_path|" "$script_path"
    fi
done

