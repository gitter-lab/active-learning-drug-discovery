"""
    Runs an active learning simulation for sample dataset.
    
    Usage:
        python sample_data_runner.py \
        --pipeline_params_json_file=../param_configs/sample_data_config.json \
        --hyperparams_json_file=../param_configs/experiment_pstp_hyperparams/sampled_hyparams/ClusterBasedWCSelector_609.json \
        --iter_max=5 \ 
        --no-precompute_dissimilarity_matrix \
        --initial_dataset_file=../datasets/sample_data/training_data/iter_0.csv.gz
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
    
stored_hash = "eea9cbf3b56bfbeb2b0186e0d0b4a43f7cd1abd32dc99c234b3eb47048012f9c"
  
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_params_json_file', action="store", dest="pipeline_params_json_file", required=True)
    parser.add_argument('--hyperparams_json_file', action="store", dest="hyperparams_json_file", required=True)
    parser.add_argument('--iter_max', type=int, default=5, action="store", dest="iter_max", required=True)
    parser.add_argument('--precompute_dissimilarity_matrix', dest='precompute_dissimilarity_matrix', action='store_true')
    parser.add_argument('--no-precompute_dissimilarity_matrix', dest='precompute_dissimilarity_matrix', action='store_false')
    parser.add_argument('--initial_dataset_file', action="store", dest="initial_dataset_file", default=None, required=True)
    
    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    hyperparams_json_file = given_args.hyperparams_json_file
    initial_dataset_file = given_args.initial_dataset_file
    iter_max = given_args.iter_max
    precompute_dissimilarity_matrix = given_args.precompute_dissimilarity_matrix
    
    ##########################################################################################################################
    # load param json configs
    # 1. pipeline config contains general parameters for the experiment: training data info, unlabeled data info, model data
    # 2. hyperparams config contains parameters for the specific next batch selector/strategy 
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    with open(hyperparams_json_file) as f:
        next_batch_selector_params = json.load(f)
    
    train_files = glob.glob(pipeline_config['training_data_params']['data_path_format'].format('*'))
    [os.remove(x) for x in train_files if 'iter_0.csv.gz' not in x]
    
    ##########################################################################################################################
    next_batch_csv_fmt = pipeline_config['common']['next_batch_csv']
    
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
    
    ##########################################################################################################################
    for iter_num in range(iter_max):
        # run current iter for this simulation
        iter_start_time = time.time()
        print('---------------------------------------------------------------')
        print('Processing iteration number: {}...'.format(iter_num))
        #### Run single iteration of active learning pipeline ####
        selection_start_time = time.time()
        
        # see get_next_batch in active_learning_dd.active_learning_dd to understand how the codebase processes an iteration
        exploitation_df, exploration_df, exploitation_array, exploration_array = get_next_batch(training_loader_params=pipeline_config['training_data_params'], 
                                                                                                unlabeled_loader_params=pipeline_config['unlabeled_data_params'],
                                                                                                model_params=pipeline_config['model'],
                                                                                                task_names=pipeline_config['common']['task_names'],
                                                                                                next_batch_selector_params=next_batch_selector_params,
                                                                                                dissimilarity_memmap_filename=pipeline_config['common']['dissimilarity_memmap_filename'])
        selection_end_time = time.time()
        total_selection_time = selection_end_time - selection_start_time
        
        #### Save selection in csv according to pipeline config file format ####
        # save results
        if exploitation_df is not None or exploration_df is not None:
            next_batch_csv = next_batch_csv_fmt.format(iter_num+1)
            # finally save the exploitation, exploration dataframes to training data directory for next iteration
            pd.concat([exploitation_df, exploration_df]).to_csv(next_batch_csv, compression='gzip',
                                                                index=False)
        iter_end_time = time.time()
        print('Finished processing iteration {}. Took {} seconds.'.format(iter_num, iter_end_time-iter_start_time))
            
    ##########################################################################################################################
    # Check correctness by hasing the selected compounds and seeing if it matches the predefined hash
    selected_df = pd.concat([pd.read_csv(pipeline_config['training_data_params']['data_path_format'].format(i)) for i in range(1,iter_max+1)])
    import hashlib
    from pandas.util import hash_pandas_object
    
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(selected_df, index=True).values).hexdigest()
    
    assert selected_df.shape[0] == (iter_max)*batch_size 
    assert df_hash == stored_hash
    
    train_files = glob.glob(pipeline_config['training_data_params']['data_path_format'].format('*'))
    [os.remove(x) for x in train_files if 'iter_0.csv.gz' not in x]
    
    print()
    print("Finished testing sample dataset. Verified that hashed selection matches stored hash.")