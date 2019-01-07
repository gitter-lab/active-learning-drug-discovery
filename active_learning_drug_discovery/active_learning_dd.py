"""
    Contains convenience functions for running the active learning pipeline.
    Typically takes in dictionary object specifying the configs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

from active_learning_dd.models.prepare_model import prepare_model
from active_learning_dd.database_loaders.prepare_loader import prepare_loader
from active_learning_dd.next_batch_selector.prepare_selector import prepare_selector

"""
    Runs a single iteration of the active learning pipeline.
    
    Returns:
    (selected_batch_df, 
     exploitation_array, 
     exploration_array)
    
    where exploitation_array and exploration_array are of the form:
    [instance_index, cluster_id]
"""
def get_next_batch(training_loader_params, 
                   unlabeled_loader_params,
                   model_params,
                   task_names,
                   next_batch_selector_params):
    # load training data
    training_loader = prepare_loader(data_loader_params=training_loader_params,
                                     task_names=task_names)
    X_train, y_train = training_loader.get_features_and_labels()
    
    # load and train model
    model = load_model(model_params=model_params,
                       task_names=task_names)
    model.fit(X_train, y_train)
    
    # load unlabeled pool
    unlabeled_loader = prepare_loader(data_loader_params=unlabeled_loader_params,
                                      task_names=task_name)
    X_unlabeled = unlabeled_loader.get_features()
    
    # select next batch
    next_batch_selector = prepare_selector(training_loader=training_loader,
                                           unlabeled_loader=unlabeled_loader,
                                           trained_model=model,
                                           next_batch_selector_params=next_batch_selector_params)
    selected_clusters_instances_pairs = next_batch_selector.select_next_batch()
    selected_exploitation_cluster_instances_pairs = selected_clusters_instances_pairs[0]
    selected_exploration_cluster_instances_pairs = selected_clusters_instances_pairs[1]
    
    # get unlabeled dataframe slice corresponding to selected pairs
    unlabeled_df = unlabeled_loader.get_dataframe()
    exploitation_array = unroll_cluster_instances_pairs(selected_exploitation_cluster_instances_pairs)
    exploration_array = unroll_cluster_instances_pairs(selected_exploration_cluster_instances_pairs)
    exploitation_df = unlabeled_df.iloc[exploitation_array[:,0],:]
    exploration_df = unlabeled_df.iloc[exploration_array[:,0],:]
    
    return exploitation_df, exploration_df, exploitation_array, exploration_array
    
    
"""
    Unrolls cluster instance pairs list into a 2D numpy array with cols: [instance_idx, cluster_id] 
"""
def unroll_cluster_instances_pairs(cluster_instances_pairs):
    cluster_instances_pairs = a[:]
    instance_array = np.hstack([x[1] for x in cluster_instances_pairs]).reshape(-1,1)
    cluster_array = np.hstack([np.repeat(x[0], len(x[1])) for x in cluster_instances_pairs]).reshape(-1,1)
        
    instance_cluster_array = np.hstack([instance_array, cluster_array])
    return instance_cluster_array