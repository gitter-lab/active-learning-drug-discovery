"""
    Runs an active learning simulation.
    
    Usage:
        python simulation_runner.py \
        --pipeline_params_json_file=../param_configs/general_pipeline_config.json \
        --nbs_params_json_file=../param_configs/ClusterBasedWCSelector_params_test.json \
        --iter_max=5 \ 
        --process_num=$process_num
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

from active_learning_dd.active_learning_dd import get_next_batch
from active_learning_dd.database_loaders.prepare_loader import prepare_loader
from active_learning_dd.utils.data_utils import get_duplicate_smiles_in1d
from simulation_utils import *
    
    
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_params_json_file', action="store", dest="pipeline_params_json_file", required=True)
    parser.add_argument('--nbs_params_json_file', action="store", dest="nbs_params_json_file", required=True)
    parser.add_argument('--iter_max', type=int, default=10, action="store", dest="iter_max", required=False)
    parser.add_argument('--process_num', type=int, default=0, action="store", dest="process_num", required=False)
    
    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    nbs_params_json_file = given_args.nbs_params_json_file
    iter_max = given_args.iter_max
    process_num = given_args.process_num
    
    # load param json configs
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    with open(nbs_params_json_file) as f:
        nbs_config = json.load(f)

    if process_num % 2 == 0: # even process_num
        next_batch_selector_params = get_random_params(nbs_config, rnd_seed=process_num)
    else:
        next_batch_selector_params = get_param_from_dist(nbs_config, rnd_seed=process_num)
        
    params_set_results_dir = pipeline_config['common']['params_set_results_dir'].format(next_batch_selector_params['class'], process_num)
    params_set_config_csv = params_set_results_dir+'/'+pipeline_config['common']['params_set_config_csv']
    pathlib.Path(params_set_config_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(params_set_config_csv,'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerow(next_batch_selector_params.keys())
        csv_w.writerow(next_batch_selector_params.values())
    # run this param set for each batch size
    batch_size_list = next_batch_selector_params["batch_size"]
    for batch_size in batch_size_list:
        next_batch_selector_params["batch_size"] = batch_size
        batch_size_results_dir = params_set_results_dir + pipeline_config['common']['batch_size_results_dir']
        # run iterations for this simulation
        for iter_num in range(iter_max):
            iter_start_time = time.time()
            print('Processing iteration number: {}...'.format(iter_num))
            #### Run single iteration of active learning pipeline ####
            selection_start_time = time.time()
            exploitation_df, exploration_df, exploitation_array, exploration_array = get_next_batch(training_loader_params=pipeline_config['training_data_params'], 
                                                                                                    unlabeled_loader_params=pipeline_config['unlabeled_data_params'],
                                                                                                    model_params=pipeline_config['model'],
                                                                                                    task_names=pipeline_config['common']['task_names'],
                                                                                                    next_batch_selector_params=next_batch_selector_params)
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