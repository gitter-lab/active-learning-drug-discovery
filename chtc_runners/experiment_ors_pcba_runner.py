"""
    Runs a one round screening simulation for experiment 107 tasks in the PCBA dataset. 
    Initial training data was sampled from task_name dataset using uniform random sampling or diversity (Tanimoto dissimilarity) sampling.
    
    Usage:
        python experiment_ors_pcba_runner.py \
        --pipeline_params_json_file=../param_configs/experiment_pcba_ors/ors_pcba_pipeline_config.json \
        --training_data_fmt=../datasets/pcba_ors/{}/random/size_{}/sample_{}/ \
        --task_name=pcba-aid881 \
        --sampling_type=random \
        --sample_size=489 \
        --seed=55886611 \
        --max_size=4896 \
        --pcba_all_csv_file=../datasets/pcba/pcba_all.csv.gz
        
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

""" 
    Helper function to uniformly sample training set. 
    Saves train (sampled) and unlabeled (non-sampled) csv files to train_file directory. 
"""
def uniform_random_sample(task_df, train_file, seed, sample_size, task_name):
    print('Uniform random sampling from dataset...')
    start_time = time.time()
    
    dataset_size = task_df.shape[0]
    active_indices = np.where(task_df[task_name] == 1)[0]
    
    np.random.seed(seed)
    cpds_to_select = np.zeros(sample_size, dtype=int)
    cpds_to_select[0] = np.random.choice(active_indices, replace=False, size=1) # guarantee at least one hit is in training set
    cpds_to_select[1:] = np.random.choice(np.setdiff1d(np.arange(dataset_size), [cpds_to_select[0]]), replace=False, size=(sample_size-1))

    sample_df = task_df.iloc[cpds_to_select,:]
    sample_df.to_csv(train_file, compression='gzip', index=False)

    assert sample_df[task_name].sum() > 0
    assert sample_df.shape[0] == sample_size
    
    unlabeled_df = task_df.iloc[np.setdiff1d(np.arange(dataset_size), cpds_to_select)]
    unlabeled_df.to_csv(train_file.replace('train', 'unlabeled'), compression='gzip', index=False)
    
    tmp_df = pd.concat([sample_df, unlabeled_df]).sort_values('Index ID')
    assert task_df.equals(tmp_df)
    
    end_time = time.time()
    print('Uniform random sampled {} compounds from task {} with seed {}. Total time {} minutes.'.format(sample_df.shape[0], task_name, seed,
                                                                                                         (end_time - start_time)/60.0))
    

""" 
    Helper function to diversity (Tanimoto dissimilarity) sampling.
    Saves train (sampled) and unlabeled (non-sampled) csv files to train_file directory. 
"""
def diversity_sample(task_df, train_file, seed, sample_size, task_name):
    print('Diversity sampling from dataset...')
    start_time = time.time()
    
    dataset_size = task_df.shape[0]
    active_indices = np.where(task_df[task_name] == 1)[0]
    
    np.random.seed(seed)
    cpds_to_select = np.zeros(sample_size, dtype=int)
    cpds_to_select[0] = np.random.choice(active_indices, replace=False, size=1) # guarantee at least one hit is in training set
    
    X_prosp = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in task_df['Morgan FP_2_1024']]).astype(float)
    for i in range(1, sample_size):
        x = X_prosp[cpds_to_select[:i],:]
        remaining_cpds = np.setdiff1d(np.arange(X_prosp.shape[0]), cpds_to_select[:i])
        y = X_prosp[remaining_cpds,:]
        
        # adapted from: https://github.com/deepchem/deepchem/blob/2531eca8564c1dc68910d791b0bcd91fd586afb9/deepchem/trans/transformers.py#L752
        numerator = np.dot(y, x.T).flatten() # equivalent to np.bitwise_and(X_batch, Y_batch), axis=1)
        denominator = 1024 - np.dot(1-y, (1-x).T).flatten() # np.sum(np.bitwise_or(X_rep, Y_rep), axis=1)

        tandist = numerator / denominator
        tandist = 1.0 - tandist

        tandist = tandist.reshape(y.shape[0], -1)
        
        mean_dist_to_selected = tandist.mean(axis=1)
        farthest_idx = np.argmax(mean_dist_to_selected)

        cpds_to_select[i] = remaining_cpds[farthest_idx]

    sample_df = task_df.iloc[cpds_to_select,:]
    sample_df.to_csv(train_file, compression='gzip', index=False)
    
    assert sample_df[task_name].sum() > 0
    assert sample_df.shape[0] == sample_size
    
    unlabeled_df = task_df.iloc[np.setdiff1d(np.arange(dataset_size), cpds_to_select)]
    unlabeled_df.to_csv(train_file.replace('train', 'unlabeled'), compression='gzip', index=False)
    
    tmp_df = pd.concat([sample_df, unlabeled_df]).sort_values('Index ID')
    assert task_df.equals(tmp_df)
    
    end_time = time.time()
    print('Diversity sampled {} compounds from task {}. Time {} minutes.'.format(sample_df.shape[0], task_name, (end_time - start_time)/60.0))
  
  

# constants
SEEDS_LIST = [55886611, 91555713, 10912561, 69210899, 75538109, 
              33176925, 17929553, 26974345, 63185387, 54808003]
              
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_params_json_file', action="store", dest="pipeline_params_json_file", required=True)
    parser.add_argument('--training_data_fmt', default=None, action="store", dest="training_data_fmt", required=True)
    parser.add_argument('--task_name', default=None, action="store", dest="task_name", required=True)
    parser.add_argument('--sampling_type', default='random', action="store", dest="sampling_type", required=True)
    parser.add_argument('--sample_size', default=489, type=int, action="store", dest="sample_size", required=True)
    parser.add_argument('--seed', default=55886611, type=int, action="store", dest="seed", required=True)
    parser.add_argument('--max_size', default=4896, type=int, action="store", dest="max_size", required=False)
    parser.add_argument('--pcba_all_csv_file', default='../datasets/pcba/pcba_all.csv.gz', action="store", dest="pcba_all_csv_file", required=False)
    
    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    training_data_fmt = given_args.training_data_fmt
    task_name = given_args.task_name
    sampling_type = given_args.sampling_type.lower()
    sample_size = given_args.sample_size
    seed = given_args.seed
    max_size = given_args.max_size
    pcba_all_csv_file = given_args.pcba_all_csv_file
    
    if sampling_type not in ['random', 'diversity']:
        print('--sampling_type parameter should be either: random or diversity.')
        import sys
        sys.exit()
    
    ors_start_time = time.time()
    # load param json configs
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    model_params = pipeline_config['model']    
    task_names = [task_name]
    
    
    # get and clean task dataframe
    data_df = pd.read_csv(pcba_all_csv_file)
    features = data_df.columns[:6].tolist() 
    
    task_df = data_df[features+[task_name]]
    task_df = task_df[~pd.isna(task_df[task_name])]
    task_df = task_df.sort_values('Index ID')
    task_df = task_df.dropna()
    task_df = task_df.reset_index(drop=True)
    
    # sample training set
    training_data_file = training_data_fmt + '/train.csv.gz'
    training_data_file = training_data_file.format(task_name, sample_size, SEEDS_LIST.index(seed))
    pathlib.Path(training_data_file).parent.mkdir(parents=True, exist_ok=True)
    
    print('Set {} as starting initial training set.'.format(training_data_file))
    
    if sampling_type == 'random':
        uniform_random_sample(task_df, training_data_file, seed, sample_size, task_name)
    elif sampling_type == 'diversity':
        diversity_sample(task_df, training_data_file, seed, sample_size, task_name)
    
    
    print('----------------------------------------------------------------------------------------------')
    # set training and unlabeled data
    import copy
    unlabeled_loader_params = pipeline_config['data_params']
    training_loader_params = copy.deepcopy(pipeline_config['data_params'])
    unlabeled_loader_params['data_path_format'] = training_data_file.replace('train', 'unlabeled')
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
    print('----------------------------------------------------------------------------------------------')
    
    # batch_size is max_size - X_train.shape[0]
    batch_size = max_size - X_train.shape[0]
    
    # load and train model
    start_time = time.time()
    model = prepare_model(model_params=model_params,
                          task_names=task_names)
    model.fit(X_train, y_train)
    end_time = time.time()
    print('Finished training model. Took {} seconds.'.format(end_time - start_time))
    print('----------------------------------------------------------------------------------------------')
    
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
    output_file = training_data_file.replace('train', 'selected')
    selected_df.to_csv(output_file, compression='gzip', index=False)
    
    tmp_df = pd.concat([training_loader.get_dataframe(), selected_df]).drop_duplicates(subset='Index ID')
    assert tmp_df.shape[0] == max_size
    
    ors_end_time = time.time()
    print('Finished processing one round screen. Took {} seconds.'.format(ors_end_time-ors_start_time))
    print('----------------------------------------------------------------------------------------------')