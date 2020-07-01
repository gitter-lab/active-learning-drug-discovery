"""
    Runs an active learning simulation for experiment 4 - PstP.
    Experiment 4 - PstP: prospective screening of PstP target. True labels are unknown. 
    
    Usage:
        python experiment_pstp_runner.py \
        --pipeline_params_json_file=../param_configs/exp_pstp_pipeline_config.json \
        --hyperparams_json_file=../param_configs/experiment_pstp_hyperparams/custom_cbws/ClusterBasedWCSelector_custom_1.json \
        --iter_num=0 \ 
        --label_setting=adaptive \
        --no-precompute_dissimilarity_matrix \
        [--initial_dataset_file=../datasets/pstp/diverse_plate.csv]
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

from active_learning_dd.active_learning_dd import get_next_batch
from active_learning_dd.database_loaders.prepare_loader import prepare_loader
from active_learning_dd.utils.data_utils import get_duplicate_smiles_in1d
from active_learning_dd.utils.generate_dissimilarity_matrix import compute_dissimilarity_matrix
from simulation_utils import *
    
    
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_params_json_file', action="store", dest="pipeline_params_json_file", required=True)
    parser.add_argument('--hyperparams_json_file', action="store", dest="hyperparams_json_file", required=True)
    parser.add_argument('--iter_num', type=int, default=0, action="store", dest="iter_num", required=True)
    parser.add_argument('--label_setting', action="store", dest="label_setting", required=True)
    parser.add_argument('--precompute_dissimilarity_matrix', dest='precompute_dissimilarity_matrix', action='store_true')
    parser.add_argument('--no-precompute_dissimilarity_matrix', dest='precompute_dissimilarity_matrix', action='store_false')
    parser.add_argument('--initial_dataset_file', action="store", dest="initial_dataset_file", default=None, required=False)
    
    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    hyperparams_json_file = given_args.hyperparams_json_file
    initial_dataset_file = given_args.initial_dataset_file
    iter_num = given_args.iter_num
    label_setting = given_args.label_setting
    precompute_dissimilarity_matrix = given_args.precompute_dissimilarity_matrix
    
    # load param json configs
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    with open(hyperparams_json_file) as f:
        next_batch_selector_params = json.load(f)
    
    # setup initial training plate
    pathlib.Path(pipeline_config['training_data_params']['data_path_format'].format(label_setting, 0)).parent.mkdir(parents=True, exist_ok=True)
        
    if initial_dataset_file is not None:
        shutil.copyfile(initial_dataset_file, pipeline_config['training_data_params']['data_path_format'].format(label_setting, 0))
        assert pd.read_csv(initial_dataset_file).equals(pd.read_csv(pipeline_config['training_data_params']['data_path_format'].format(0)))
    else:
        initial_dataset_file = 'None' 
        
    print('Set {} as starting initial set.'.format(initial_dataset_file))
    
    # confirm cumulative_${iter_num}.csv.gz exists
    current_train_csv = pipeline_config['training_data_params']['data_path_format'].format(label_setting, iter_num)
    pipeline_config['training_data_params']['data_path_format'] = current_train_csv
    if (iter_num > 0) or (iter_num == 0 and initial_dataset_file != 'None'):
        assert os.path.exists(current_train_csv)
    
    next_batch_csv = pipeline_config['common']['next_batch_csv'].format(label_setting, iter_num)
    
    # set target/task name
    if label_setting == "adaptive":
        pipeline_config['common']['task_names'] = pipeline_config['common']['task_names'].format("Adaptive")
    else:
        pipeline_config['common']['task_names'] = pipeline_config['common']['task_names'].format("True")
    
    # get batch size
    batch_size = pipeline_config['common']['batch_size']
    next_batch_selector_params["batch_size"] = batch_size
    print('---------------------------------------------------------------')
    print('Starting AL pipeline with batch_size: {}'.format(batch_size))
    
    try: 
        pipeline_config['common']['dissimilarity_memmap_filename']
    except:
        pipeline_config['common']['dissimilarity_memmap_filename'] = None
    
    if precompute_dissimilarity_matrix:
        if pipeline_config['common']['dissimilarity_memmap_filename'] is None:
            pipeline_config['common']['dissimilarity_memmap_filename'] = '../datasets/dissimilarity_matrix.dat'
        compute_dissimilarity_matrix(csv_file_or_dir=pipeline_config['unlabeled_data_params']['data_path_format'], 
                                     output_dir=pipeline_config['common']['dissimilarity_memmap_filename'])
    
    # run current iter for this simulation
    iter_start_time = time.time()
    print('---------------------------------------------------------------')
    print('Processing iteration number: {}...'.format(iter_num))
    #### Run single iteration of active learning pipeline ####
    selection_start_time = time.time()
    exploitation_df, exploration_df, exploitation_array, exploration_array = get_next_batch(training_loader_params=pipeline_config['training_data_params'], 
                                                                                            unlabeled_loader_params=pipeline_config['unlabeled_data_params'],
                                                                                            model_params=pipeline_config['model'],
                                                                                            task_names=pipeline_config['common']['task_names'],
                                                                                            next_batch_selector_params=next_batch_selector_params,
                                                                                            dissimilarity_memmap_filename=pipeline_config['common']['dissimilarity_memmap_filename'])
    selection_end_time = time.time()
    total_selection_time = selection_end_time - selection_start_time
    #### Evaluation: no evaluation since labels are unknown. Only save selection. ####
    # save results
    if exploitation_df is not None or exploration_df is not None:
        # finally save the exploitation, exploration dataframes to training data directory for next iteration
        pd.concat([exploitation_df, exploration_df]).to_csv(next_batch_csv, compression='gzip',
                                                            index=False)
    iter_end_time = time.time()
    print('Finished processing iteration {}. Took {} seconds.'.format(iter_num, iter_end_time-iter_start_time))

    # terminate if both exploitation and exploration df are None
    if exploitation_df is None and exploration_df is None:
        print('Both exploitation and exploration selections are empty. Terminating program.')