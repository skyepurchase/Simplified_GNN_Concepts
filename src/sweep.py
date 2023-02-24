import yaml
import argparse
import os.path as osp

from main import main

# Typing
from argparse import ArgumentParser, Namespace


def sweep(filename: str,
          dataset_name: str,
          config: dict,
          args: Namespace):
    main(filename, dataset_name, config, args)


if __name__=="__main__":
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Config file")
    parser.add_argument('--lr', nargs="+", help="The list of learning rate values to initialise the sweep")
    parser.add_argument('--wd', nargs="+", help="The list of weight decay values to initialise the sweep")
    args: Namespace = parser.parse_args()

    arg_dict = {
        'config': args.config,
        'lr': args.lr,
        'wd': args.wd,
        # Adding defaults for the sweep
        'root': "data/",
        'seed': 1337,
        'verbose': True
    }
    
    args = argparse.Namespace(**arg_dict)

    with open(osp.abspath(args.config), 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = args.config.split('/')[-1]
        dataset_name = filename.split('.')[2]
        sweep(filename, dataset_name, config, args)

