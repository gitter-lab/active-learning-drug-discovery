"""
    Utility functions for simulations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
from sklearn.model_selection import ParameterGrid
import pathlib
import numpy as np
import pandas as pd

"""
    Helper function to return max hits and max cluster hits from the 
    unlabeled data. Note this is used in the simulation since in a real
    scenario these values are unknown.
"""
def get_unlabeled_maxes(training_loader_params, 
                        unlabeled_loader_params,
                        task_names,
                        batch_size):
    # load loaders
    training_loader = prepare_loader(data_loader_params=training_loader_params,
                                     task_names=task_names)
    unlabeled_loader = prepare_loader(data_loader_params=unlabeled_loader_params,
                                      task_names=task_names)
    
    # remove already labeled molecules by checking training and unlabeled pool overlap
    # note duplicates determined via rdkit smiles
    smiles_train = training_loader.get_smiles()
    smiles_unlabeled = unlabeled_loader.get_smiles()
    idx_to_drop = get_duplicate_smiles_in1d(smiles_train, smiles_unlabeled)
    unlabeled_loader.idx_to_drop = idx_to_drop
    
    # now get labels and clusters
    y_unlabeled = unlabeled_loader.get_labels()
    y_clusters = unlabeled_loader.get_clusters()
    
    max_hits_list = np.sum(y_unlabeled, axis=0)
    max_hits_list = [min(batch_size, actives_count) for actives_count in max_hits_list]
    
    max_cluster_hits_list = [0 for _ in range(len(task_names))]
    for ti in range(len(task_names)):
        # Get the clusters with actives
        active_indices = np.where(y_unlabeled[:,ti] == 1)[0]
        clusters_with_actives_ti = y_clusters[active_indices]
        max_cluster_hits_list[ti] = min(batch_size, 
                                        np.unique(clusters_with_actives_ti).shape[0])
    
    return max_hits_list, max_cluster_hits_list
    
    
"""
    Random sample from the given parameter set. 
    Assumes uniform distribution.
"""
def get_random_params(nbs_config,
                      rnd_seed=0):
    # pop the batch_size, since we want to simulate all batch sizes for this param set
    next_batch_selector_params = nbs_config["next_batch_selector_params"]
    batch_sizes = next_batch_selector_params.pop("batch_size", None)
    # sample random param 
    param_grid = ParameterGrid(next_batch_selector_params)
    np.random.seed(rnd_seed)
    param_idx = np.random.randint(len(param_grid), size=1)[0]
    next_batch_selector_params = param_grid[param_idx]
    next_batch_selector_params["batch_size"] = batch_sizes
    return next_batch_selector_params
    
"""
    Random sample from the given parameter set using the 
    distribution given in the config file.
"""
def get_param_from_dist(nbs_config,
                        rnd_seed=0):
    nbs_params = nbs_config["next_batch_selector_params"]
    nbs_params_probas = nbs_config["nbs_params_probas"]
    # sample random param 
    np.random.seed(rnd_seed)
    for param in nbs_params_probas:
        param_choices = nbs_params[param]
        param_probas = nbs_params_probas[param]
        param_sampled_choice = np.random.choice(param_choices, size=1, p=param_probas)[0]
        
        # modify nbs_params dict with sampled choice
        nbs_params[param] = param_sampled_choice
    
    return nbs_params