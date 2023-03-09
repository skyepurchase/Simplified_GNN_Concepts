from io import TextIOWrapper
from datetime import datetime
import numpy as np

import os.path as osp
from os import mkdir

from matplotlib import cm
import matplotlib.pyplot as plt

from main import main

# Typing
from argparse import ArgumentParser, Namespace
from typing import List, Tuple
from numpy.typing import NDArray


DIR = osp.dirname(__file__)


def sweep(filename: str,
          dataset_name: str,
          save_name: str,
          config: dict,
          args: Namespace):
    """A wrapper for the main train function to allow insertion of specific hyperparameters into the config
    INPUT
        filename        : The file name for storing the different training results
        dataset_name    : The name of the dataset used during training and testing
        save_name       : The name of the experiment group to save results to
        config          : The configuration of the model being trained
        args            : The Namespace argument of passed arguments (such as learning rate)
    """
    save_path: str = osp.join(DIR, "../output", save_name)

    if not osp.exists(save_path):
        mkdir(save_path)

    results: List[List[float]] = []
    time = datetime.now()
    results_file: TextIOWrapper = open(osp.join(save_path, f'lr-{args.lr}_wd-{args.wd}_{time.strftime("%m%d-%H%M%S")}.csv'), "w")

    learning_rates = args.lr
    weight_decays = []

    wrapper_kwargs = config["wrapper"]["kwargs"]

    if "weight_decay" in wrapper_kwargs:
        if args.wd is None:
            raise ValueError("Provided config uses weight decay but no weight decay values provided.")
        else:
            weight_decays = args.wd

    # (accuracy, hyperparameter)
    best_lr: Tuple[float, float] = (0.0, 0.0)
    best_wd: Tuple[float, float] = (0.0, 0.0)

    for learning_rate in learning_rates:
        config["wrapper"]["kwargs"]["learning_rate"] = learning_rate

        if "weight_decay" in wrapper_kwargs:
            for weight_decay in weight_decays:
                print("hyperparameters", learning_rate, weight_decay)

                config["wrapper"]["kwargs"]["weight_decay"] = weight_decay
                accuracy = main(filename, dataset_name, config, args)

                results.append([learning_rate, weight_decay, accuracy])

                print("accuracy", accuracy)

                if accuracy > best_wd[1]:
                    best_wd = (accuracy, weight_decay)

                if accuracy > best_lr[1]:
                    best_lr = (accuracy, learning_rate)
        else:
            print("hyperparameters", learning_rate)

            accuracy = main(filename, dataset_name, config, args)
            print("accuracy", accuracy)

            results.append([learning_rate, accuracy])

            if accuracy > best_lr[1]:
                best_lr = (accuracy, learning_rate)

    np_res: NDArray = np.array(results)
    np.savetxt(results_file, np_res, delimiter=',')

    # Creating a hyperparameter surface or plot to help visualise the space
    if "weight_decay" in wrapper_kwargs:
        num_lrs: int = len(args.lr)
        num_wds: int  = len(args.wd)

        grid_Z: NDArray = np.zeros((num_lrs, num_wds))

        for i in range(len(args.lr)):
            grid_Z[i,:] = np_res[num_wds*i:num_wds*(i+1), 2]

        grid_X: NDArray
        grid_Y: NDArray
        grid_X, grid_Y = np.meshgrid(np.array(args.lr), np.array(args.wd))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)

        fig.suptitle(f"Hyperparameter surface for {save_name}")
        ax.plot_surface(
            np.log10(grid_X),
            np.log10(grid_Y),
            grid_Z.T,
            linewidth=0,
            cmap=cm.coolwarm,
            antialiased=False
        )
        ax.set_xlabel(f"Log learning rate from {args.lr}")
        ax.set_ylabel(f"Log weight decay from {args.wd}")
        ax.set_zlabel(f'Accuracy after {config["trainer"]["max_epochs"]} epochs')
        ax.set_zlim(0,100)
    else:
        fig, ax = plt.subplots(dpi=200)

        fig.suptitle(f"Hyperparameter surface for {save_name}")
        ax.plot(np.log10(np_res[:,0]), np_res[:,1])
        ax.set_xlabel(f"Log learning rate from {args.lr}")
        ax.set_ylabel(f'Accuracy after {config["trainer"]["max_epochs"]} epochs')
        ax.set_ylim(0,100)
        
    plt.savefig(osp.join(save_path, f'lr-{args.lr}_wd-{args.wd}_{time.strftime("%m%d-%H%M%S")}.png'))


if __name__=="__main__":
    import yaml
    import logging
    import argparse

    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Config file")
    parser.add_argument('--lr', type=float, nargs="+", required=True, help="The learning rate value test")
    parser.add_argument('--wd', type=float, nargs="+", help="The weight decay value to test")
    args: Namespace = parser.parse_args()

    arg_dict = {
        'config': args.config,
        'lr': args.lr,
        'wd': args.wd,
        # Adding defaults for the sweep
        'root': "data/",
        'seed': 1337,
        'verbose': False
    }
    
    args = argparse.Namespace(**arg_dict)

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    with open(osp.abspath(args.config), 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = args.config.split('/')[-1]
        dataset_name = filename.split('.')[2]
        save_name = filename.split('.')[0] + "-" + dataset_name
        sweep(filename, dataset_name, save_name, config, args)

