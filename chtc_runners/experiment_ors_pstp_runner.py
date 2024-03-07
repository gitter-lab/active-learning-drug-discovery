"""
    Runs a one round screening simulation for experiment 4 - PstP. 
    Initial training data was sampled from PstP dataset using uniform random sampling or diversity (Tanimoto dissimilarity) sampling.
    Experiment 4 - PstP: prospective screening of PstP target. 
    
    NOTE: This dataset was generated from notebook Experiment 4 - One Round Screening - Prepare Datasets
    
    Usage:
        python experiment_ors_pstp_runner.py \
        --pipeline_params_json_file=../param_configs/experiment_pstp_hyperparams/one_round_screening/ors_pstp_pipeline_config.json \
        --training_data_dir=../datasets/pstp/one_round_screening/random/size_400/sample_0/ \ 
        --max_size=4000
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
    
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_params_json_file', action="store", dest="pipeline_params_json_file", required=True)
    parser.add_argument('--training_data_dir', default=None, action="store", dest="training_data_dir", required=True)
    parser.add_argument('--max_size', default=4000, type=int, action="store", dest="max_size", required=False)
    
    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    training_data_dir = given_args.training_data_dir
    max_size = given_args.max_size
    
    ors_start_time = time.time()
    # load param json configs
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    model_params = pipeline_config['model']
    task_names = pipeline_config['task_names']
    
    training_data_file = training_data_dir + '/train.csv.gz'
    print('Set {} as starting initial training set.'.format(training_data_file))
    
    # load training and unlabeled data
    import copy
    unlabeled_loader_params = pipeline_config['data_params']
    training_loader_params = copy.deepcopy(pipeline_config['data_params'])
    training_loader_params['data_path_format'] = training_data_file
    
    start_time = time.time()
    training_loader = prepare_loader(data_loader_params=training_loader_params,
                                     task_names=task_names)
    unlabeled_loader = prepare_loader(data_loader_params=unlabeled_loader_params,
                                      task_names=task_names)
    # remove training data from unlabeled pool dataset
    unlabeled_loader.drop_duplicates_via_smiles(training_loader.get_smiles())

    X_train, y_train = training_loader.get_features_and_labels()
    X_unlabeled = unlabeled_loader.get_features()
    end_time = time.time()
    print('Finished loading data. Took {} seconds.'.format(end_time - start_time))
    print('Training data shape X_train: {}, y_train: {}'.format(X_train.shape, y_train.shape))
    print('Unlabeled data shape X_unlabeled: {}'.format(X_unlabeled.shape))
    
    # batch_size is max_size - X_train.shape[0]
    batch_size = max_size - X_train.shape[0]
    
    # load and train model
    start_time = time.time()
    model = prepare_model(model_params=model_params,
                          task_names=task_names)
    model.fit(X_train, y_train)
    end_time = time.time()
    print('Finished training model. Took {} seconds.'.format(end_time - start_time))
    
    # predict on unlabeled pool
    preds_unlabeled = model.predict(unlabeled_loader.get_features())[:,0] 
    
    # select top batch_size predicted instances
    selection_start_time = time.time()
    
    top_predicted_idx = np.argsort(preds_unlabeled)[::-1][:batch_size]
    unlabeled_df = unlabeled_loader.get_dataframe()
    selected_df = unlabeled_df.iloc[top_predicted_idx,:]
    
    selection_end_time = time.time()
    total_selection_time = selection_end_time - selection_start_time
    
    # save results
    output_file = training_data_dir + '/selected.csv.gz'
    selected_df.to_csv(output_file, compression='gzip', index=False)
    
    ors_end_time = time.time()
    print('Finished processing one round screen. Took {} seconds.'.format(ors_end_time-ors_start_time))
    
    
    print('train: {}, {} ...... test:{}, {}'.format(y_train.shape[0], y_train.sum(), selected_df.shape[0], selected_df['PstP True Active'].sum()))