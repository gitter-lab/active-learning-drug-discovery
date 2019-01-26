"""
    Runs an active learning simulation.
    
    Usage:
        python simulation_runner.py \
        --pipeline_params_json_file=../param_configs/general_pipeline_config.json \
        --nbs_params_json_file=../param_configs/ClusterBasedWCSelector_params_test.json \
        --iter_max=5 \ 
        --process_num=$process_num \
        --random_param=False
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
from active_learning_dd.utils.evaluation import evaluate_selected_batch
from simulation_utils import *
    
    
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_params_json_file', action="store", dest="pipeline_params_json_file", required=True)
    parser.add_argument('--nbs_params_json_file', action="store", dest="nbs_params_json_file", required=True)
    parser.add_argument('--iter_max', type=int, default=10, action="store", dest="iter_max", required=False)
    parser.add_argument('--process_num', type=int, default=0, action="store", dest="process_num", required=False)
    parser.add_argument('--random_param', type=bool, default=False, action="store", dest="random_param", required=False)
    
    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    nbs_params_json_file = given_args.nbs_params_json_file
    iter_max = given_args.iter_max
    process_num = given_args.process_num
    random_param = given_args.random_param
    
    # load param json configs
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    with open(nbs_params_json_file) as f:
        nbs_config = json.load(f)
    
    if random_param:
        next_batch_selector_params = get_random_params(nbs_config, rnd_seed=process_num)
    else:
        next_batch_selector_params = get_param_from_dist(nbs_config, rnd_seed=process_num)
    
    w_novelty = pipeline_config['common']['metrics_params']['w_novelty']
    task_names = pipeline_config['common']['task_names']
    if not isinstance(task_names, list):
        task_names = [task_names]
    params_set_results_dir = pipeline_config['common']['params_set_results_dir'].format(next_batch_selector_params['class'], process_num)
    params_set_config_csv = params_set_results_dir+'/'+pipeline_config['common']['params_set_config_csv']
    pathlib.Path(params_set_config_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(params_set_config_csv,'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerow(next_batch_selector_params.keys())
        csv_w.writerow(next_batch_selector_params.values())
    
    # run iterations for this simulation
    total_time = 0
    for iter_num in range(iter_max):
        print('Processing iteration number: {}...'.format(iter_num))
        # run single iteration of active learning pipeline
        exploitation_df, exploration_df, exploitation_array, exploration_array = get_next_batch(training_loader_params=pipeline_config['training_data_params'], 
                                                                                                unlabeled_loader_params=pipeline_config['unlabeled_data_params'],
                                                                                                model_params=pipeline_config['model'],
                                                                                                task_names=pipeline_config['common']['task_names'],
                                                                                                next_batch_selector_params=next_batch_selector_params)
        # save results
        start_time = time.time()
        iter_results_dir = params_set_results_dir+'/'+pipeline_config['common']['iter_results_dir'].format(iter_num)
        eval_dest_file = iter_results_dir+'/'+pipeline_config['common']['eval_dest_file']
        pathlib.Path(eval_dest_file).parent.mkdir(parents=True, exist_ok=True)
        
        # retrieve max_hits_list, max_cluster_hits_list of the unlabeled data for this iteration
        max_hits_list, max_cluster_hits_list = get_unlabeled_maxes(training_loader_params=pipeline_config['training_data_params'], 
                                                                   unlabeled_loader_params=pipeline_config['unlabeled_data_params'],
                                                                   task_names=task_names,
                                                                   batch_size=next_batch_selector_params['batch_size'])
        
        if exploitation_df is not None:
            exploitation_df.to_csv(iter_results_dir+'/'+pipeline_config['common']['batch_csv'].format('exploitation'),
                                   index=False)
            y_true = exploitation_df[task_names].values
            clusters = exploitation_array[:,1]
            evaluate_selected_batch(y_true, clusters,
                                    max_hits_list, max_cluster_hits_list,
                                    task_names, eval_dest_file.format('exploitation'),
                                    add_mean_medians=True, w_novelty=w_novelty)
        if exploration_df is not None:
            exploration_df.to_csv(iter_results_dir+'/'+pipeline_config['common']['batch_csv'].format('exploration'),
                                  index=False)
            y_true = exploration_df[task_names].values
            clusters = exploration_array[:,1]
            evaluate_selected_batch(y_true, clusters,
                                    max_hits_list, max_cluster_hits_list,
                                    task_names, eval_dest_file.format('exploration'),
                                    add_mean_medians=True, w_novelty=w_novelty)
        
        if exploitation_df is not None or exploration_df is not None:
            # finally save the exploitation, exploration dataframes to training data directory for next iteration
            pd.concat([exploitation_df, exploration_df]).to_csv(pipeline_config['training_data_params']['data_path_format'].format(iter_num+1),
                                                                index=False)
        end_time = time.time()
        print('Time it took to evaluate batch {} seconds.'.format(end_time-start_time))
        total_time += (end_time-start_time)
        print('Finished processing iteration {}. Took {} seconds.'.format(iter_num, total_time))
        
        # terminate if both exploitation and exploration df are None
        if exploitation_df is None and exploration_df is None:
            print('Both exploitation and exploration selections are empty. Terminating program.')
            break