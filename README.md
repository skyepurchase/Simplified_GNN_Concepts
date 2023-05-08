# Simplified_GNN_Concepts
The code and paper for my Part II dissertation looking into extracting concepts from simplified graph neural networks.

## Requirements

The requirements files is located in the root directory. To install the requirements navigate to the root folder, create a python virtual environment and run 
```
pip install -r requirements.txt
```

**Warning**
PyTorch lightning 1.19 was used during development and PyTorch Lighnting 2.0+ removes support for important wrapper functionality 

## Replicating results

### Training a GNN model

This is the model that will be later explained. No pre-trained modesl are provided in this directory. Individual models can be trained using the `src/main.py` function

```
python3 src/main.py -v --config <config_file> --seed <seed> --root <dataset_root_directory>
```

The configs used for the dissertation are presented in `src/configs/curr`.

To prevent having to run multiple scripts `scripts/mk_expr.sh` or `scripts/mk_all.sh` may be run to generate experiment runs. `mk_all.sh` will generate a script that runs 10 experiments to get a sample of accuracies, these use the same random seeds as those used in the dissertation.

```
./scripts/mk_expr.sh [list_config_filepaths]
```

A resulting `run/[all_]<model>-<dataset>.sh` file will be created

The multiple results from running a `run/all_<model>-<dataset>.sh` will be saved in the folder `output/<model>-<dataset>/` folder with the filename `<data>-<time>`.

### Evaluating concepts

Once a model is trained it will output an activation pickle file in `activations/`. These will follow the general naming convention `activations/<expr_name>_<seed>-<date>-<time>[_<option>].pkl`.

The concepts for a trained model can therefore be extracted, concept scores calculated, and the visualisation saved to the folder `output/<model>-<dataset>/`. Extracting and evaluating concepts requires the number of clusters, the number of hops, and the number of graphs to be visualised.

```
python3 src/eval.py --activation activations/<activation>.pkl --clusters <clusters> --hops <hops> --num_graphs <num_graphs_visualised>
```

### Adjusted Mutual Information similarity

Two activations can be compared to identified at which layer two models have the closest latent space and thus the best place to join the two models together.

```
python3 src/similarity.py --activations [list_of_activations] --clusters <clusters> --hops <hops>
```

## Building your own models

### Config files

The config files are written in YAML and follow the following basic structure

```
model:
  name: <model_name>
  kwargs: <required_kwargs>

sampler:
  name: <loader_name>

wrapper:
  name: <wrapper_name>
  kwargs: <required_kwargs>

trainer:
  dir: <log_directory>
  accelerator: <accelerator>
  devices: <num_devices>
  max_epochs: <num_epochs_run>
```

Examples for specific models, datasets, wrappers and loaders are all available in `src/configs/curr` using the naming convention `<model_name>.<loader_type>.<dataset>.<version>.yml`.

### Hyperparameter sweep

It is also possible to carry out hyperparameter sweeps to tune the designed models or verify the hyperparameter surfaces in Appendix B.

```
python3 src/sweep.py --config <config_file> --lr [learning_rates] [--wd [weight_decay_constants]]
```

## Unit tests

Most folders contain a `tests` subfolder including the relevant tests to check that the implementation is working as intended. Extensions or alterations of the code should run these tests to verify that relevant modules and functions operate as intended.

```
python3 -m unittest src.<folder>.tests.<test> -v
```
