
"""
    Runs a one round screening simulation for experiment 4 - PstP. 
    Initial training data was sampled from PstP dataset using uniform random sampling or diversity (Tanimoto dissimilarity) sampling.
    Experiment 4 - PstP: prospective screening of PstP target. 
    
    NOTE: This dataset was generated from notebook Experiment 4 - One Round Screening - Prepare Datasets

    This script runs all the one round screening simulations for the Experiment 4 - One Round Screenings
    
    Usage:
        python chtc_runners/run_all_ors_pstp_runner.py ^
                            --directory datasets/PstP/one_round_screening/ ^
                            --param_config param_configs/experiment_PstP_hyperparams/one_round_screening/ors_pstp_pipeline_config.json ^
                            --max_size 4000


        python chtc_runners/run_all_ors_pstp_runner.py --directory datasets/PstP/one_round_screening/ --param_config param_configs/experiment_PstP_hyperparams/one_round_screening/ors_pstp_pipeline_config.json --max_size 4000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import pathlib
import numpy as np
import pandas as pd
import csv 
import time
import os
import shutil

from active_learning_dd.models.prepare_model import prepare_model
from active_learning_dd.database_loaders.prepare_loader import prepare_loader

import subprocess


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='../datasets/PstP/one_round_screening/')
    parser.add_argument('--param_config', type=str, default='param_configs/experiment_PstP_hyperparams/one_round_screening/ors_pstp_pipeline_config.json')
    parser.add_argument('--max_size', type=int, default=4000)
    parser.add_argument('--regenerate', action='store_true')

    args = parser.parse_args()

    for subdir in os.listdir(args.directory):

        if(subdir == 'diversity'):
            for size_dir in os.listdir(os.path.join(args.directory, subdir)):
                print(f'Sample Size: {size_dir}')
                for sample_no in os.listdir(os.path.join(args.directory, subdir, size_dir)):
                    print(f'Sample Number: {sample_no}')
                    training_file = f'{args.directory}/{subdir}/{size_dir}/{sample_no}/'

                    if(os.path.exists(f'{args.directory}/{subdir}/{size_dir}/{sample_no}/selected.csv.gz') and not args.regenerate):
                        print(f'Already parsed through {size_dir}/{sample_no}. Continuing. ')
                        continue

                    result = subprocess.run(
                        [
                            "python", "chtc_runners/experiment_ors_pstp_runner.py",
                            f"--pipeline_params_json_file={args.param_config}",
                            f"--training_data_dir={training_file}",
                            "--max_size=4000",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        check=True,
                    )
                    print(result.stdout)
                    print('+'*50)
                print('='*50)
                

        
        elif(subdir == 'random'):
            for size_dir in os.listdir(os.path.join(args.directory, subdir)):
                print(f'Sample Size: {size_dir}')
                for sample_no in os.listdir(os.path.join(args.directory, subdir, size_dir)):
                    print(f'Sample Number: {sample_no}')
                    training_file = f'{args.directory}/{subdir}/{size_dir}/{sample_no}/'

                    if(os.path.exists(f'{args.directory}/{subdir}/{size_dir}/{sample_no}/selected.csv.gz')):
                        print(f'Already parsed through {size_dir}/{sample_no}. Continuing. ')
                        continue

                    result = subprocess.run(
                        [
                            "python", "chtc_runners/experiment_ors_pstp_runner.py",
                            f"--pipeline_params_json_file={args.param_config}",
                            f"--training_data_dir={training_file}",
                            "--max_size=4000",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        check=True,
                    )
                    print(result.stdout)
                    print('+'*50)
                print('='*50)

        else:
            raise ValueError('Error: The parent directory does not have the necessary subdirectories. Run the Experiment 4 - One Round Screening Notebook first.')