"""
    Runs an active learning simulation for experiment 1 and 2.
    Experiment 1: 10 random initial starting sets for aid624173 target for 10 iters. Prune to top-performing hyperparameters.
    Experiment 2: 10 random initial starting sets for aid624173 target for 50 iters. Further prune for experiment 3.
    
    Usage:
        python experiment_1_and_2_runner.py \
        --pipeline_params_json_file=../param_configs/exp_1_and_2_pipeline_config.json \
        --hyperparams_json_file=../param_configs/first_pass_hyperparams/middle/batch_size_96/ClusterBasedWCSelector_55.json \
        --initial_dataset_file=../datasets/aid624173_cv_96/unlabeled_1338.csv
        --iter_max=3 \ 
        --batch_size_index=0 \
        --no-precompute_dissimilarity_matrix
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
    parser.add_argument('--initial_dataset_file', action="store", dest="initial_dataset_file", required=True)
    parser.add_argument('--iter_max', type=int, default=10, action="store", dest="iter_max", required=False)
    parser.add_argument('--batch_size_index', type=int, default=0, action="store", dest="batch_size_index", required=False)
    parser.add_argument('--precompute_dissimilarity_matrix', dest='precompute_dissimilarity_matrix', action='store_true')
    parser.add_argument('--no-precompute_dissimilarity_matrix', dest='precompute_dissimilarity_matrix', action='store_false')

    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    hyperparams_json_file = given_args.hyperparams_json_file
    initial_dataset_file = given_args.initial_dataset_file
    iter_max = given_args.iter_max
    batch_size_index = given_args.batch_size_index
    precompute_dissimilarity_matrix = given_args.precompute_dissimilarity_matrix
    start_iter = 0
    
    # load param json configs
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    with open(hyperparams_json_file) as f:
        next_batch_selector_params = json.load(f)
    
    # setup initial training plate
    pathlib.Path(pipeline_config['training_data_params']['data_path_format']).parent.mkdir(parents=True, exist_ok=True)
    [os.remove(f) for f in glob.glob(pipeline_config['training_data_params']['data_path_format'].format('*'))]
    shutil.copyfile(initial_dataset_file, pipeline_config['training_data_params']['data_path_format'].format(0))
    print('Set {} as starting initial set.'.format(initial_dataset_file))
    assert pd.read_csv(initial_dataset_file).equals(pd.read_csv(pipeline_config['training_data_params']['data_path_format'].format(0)))
    
    rnd_seed = next_batch_selector_params['rnd_seed']
    
    params_set_results_dir = pipeline_config['common']['params_set_results_dir'].format(next_batch_selector_params['hyperparameter_group'],
                                                                                        next_batch_selector_params['hyperparameter_id'],
                                                                                        initial_dataset_file.split('_')[-1].replace('.csv',''))
    params_set_config_csv = params_set_results_dir+'/'+pipeline_config['common']['params_set_config_csv']
    pathlib.Path(params_set_config_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(params_set_config_csv,'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerow(list(next_batch_selector_params.keys()) + ['rnd_seed'])
        csv_w.writerow(list(next_batch_selector_params.values()) + [rnd_seed])
    # run this param set for each batch size
    batch_size_list = next_batch_selector_params["batch_size"]
    batch_size = batch_size_list[batch_size_index]
    print('---------------------------------------------------------------')
    print('Starting AL pipeline with batch_size: {}'.format(batch_size))
    next_batch_selector_params["batch_size"] = batch_size
    batch_size_results_dir = params_set_results_dir + pipeline_config['common']['batch_size_results_dir'].format(batch_size)
    
    if not os.path.exists(batch_size_results_dir):
        pathlib.Path(batch_size_results_dir+'/'+pipeline_config['common']['params_set_config_csv']).parent.mkdir(parents=True, exist_ok=True)
    
    start_iter = len(glob.glob(batch_size_results_dir + '/training_data/iter_*')) - 1    
    # modify location of training data to be able to continue jobs
    if not os.path.exists(batch_size_results_dir + '/training_data/'):
        import shutil
        shutil.copytree(pathlib.Path(pipeline_config['training_data_params']['data_path_format']).parent, 
                        batch_size_results_dir + '/training_data')
        start_iter = 0
        
    pipeline_config['training_data_params']['data_path_format'] = batch_size_results_dir + '/training_data/iter_{}.csv'
        
    with open(batch_size_results_dir+'/'+pipeline_config['common']['params_set_config_csv'],'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerow(list(next_batch_selector_params.keys()) + ['rnd_seed'])
        csv_w.writerow(list(next_batch_selector_params.values()) + [rnd_seed])
        
    try: 
        pipeline_config['common']['dissimilarity_memmap_filename']
    except:
        pipeline_config['common']['dissimilarity_memmap_filename'] = None
    
    if precompute_dissimilarity_matrix:
        if pipeline_config['common']['dissimilarity_memmap_filename'] is None:
            pipeline_config['common']['dissimilarity_memmap_filename'] = '../datasets/dissimilarity_matrix.dat'
        compute_dissimilarity_matrix(csv_file_or_dir=pipeline_config['unlabeled_data_params']['data_path_format'], 
                                     output_dir=pipeline_config['common']['dissimilarity_memmap_filename'])
    
    # run iterations for this simulation
    for iter_num in range(start_iter, min(start_iter+5, iter_max)):
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
        #### Evaluation ####
        # save results
        print('Evaluating selected batch...')
        eval_start_time = time.time()
        evaluate_selected_batch(exploitation_df, exploration_df, 
                                exploitation_array, exploration_array,
                                batch_size_results_dir,
                                pipeline_config,
                                iter_num,
                                batch_size,
                                total_selection_time,
                                add_mean_medians=False)
        eval_end_time = time.time()
        print('Time it took to evaluate batch {} seconds.'.format(eval_end_time-eval_start_time))
        
        if exploitation_df is not None or exploration_df is not None:
            # finally save the exploitation, exploration dataframes to training data directory for next iteration
            pd.concat([exploitation_df, exploration_df]).to_csv(pipeline_config['training_data_params']['data_path_format'].format(iter_num+1),
                                                                index=False)
        iter_end_time = time.time()
        print('Finished processing iteration {}. Took {} seconds.'.format(iter_num, iter_end_time-iter_start_time))

        # terminate if both exploitation and exploration df are None
        if exploitation_df is None and exploration_df is None:
            print('Both exploitation and exploration selections are empty. Terminating program.')
            break
            
    # summarize the evaluation results into a single csv file
    summarize_simulation(batch_size_results_dir,
                         pipeline_config)