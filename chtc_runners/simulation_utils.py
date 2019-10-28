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
import glob
import os

from active_learning_dd.utils.evaluation import eval_on_metrics
from active_learning_dd.database_loaders.prepare_loader import prepare_loader

"""
    Helper function to return max hits and max cluster hits from the 
    unlabeled data. Note this is used in the simulation since in a real
    scenario these values are unknown.
"""
def get_unlabeled_maxes(training_loader_params, 
                        unlabeled_loader_params,
                        task_names,
                        batch_size):
    if not isinstance(task_names, list):
        task_names = [task_names]
    # load loaders
    training_loader = prepare_loader(data_loader_params=training_loader_params,
                                     task_names=task_names)
    unlabeled_loader = prepare_loader(data_loader_params=unlabeled_loader_params,
                                      task_names=task_names)
    # remove already labeled data
    unlabeled_loader.drop_duplicates_via_smiles(training_loader.get_smiles())

    # now get labels and clusters
    y_unlabeled = unlabeled_loader.get_labels()
    unlabeled_clusters = unlabeled_loader.get_clusters()
    training_clusters = training_loader.get_clusters()

    max_hits_list = np.sum(y_unlabeled, axis=0)
    max_hits_list = [min(batch_size, actives_count) for actives_count in max_hits_list]

    max_cluster_hits_list = [0 for _ in range(len(task_names))]
    max_novel_hits_list = [0 for _ in range(len(task_names))]
    for ti in range(len(task_names)):
        # Get the clusters with actives
        active_indices = np.where(y_unlabeled[:,ti] == 1)[0]
        clusters_with_actives_ti = unlabeled_clusters[active_indices]
        unique_clusters_with_actives_ti = np.unique(clusters_with_actives_ti)
        max_cluster_hits_list[ti] = min(batch_size, 
                                        unique_clusters_with_actives_ti.shape[0])
        
        novel_clusters_with_actives = np.setdiff1d(unique_clusters_with_actives_ti, 
                                                   training_clusters)
        max_novel_hits_list[ti] = min(batch_size, 
                                      novel_clusters_with_actives.shape[0])
    return max_hits_list, max_cluster_hits_list, max_novel_hits_list

"""
    Random sample from the given parameter set. 
    Assumes uniform distribution. Samples index in the range [0, total_num_parameter_sets].
"""
def get_random_params_int_based(nbs_config,
                                rnd_seed=0):
    # pop the batch_size, since we want to simulate all batch sizes for this param set
    next_batch_selector_params = nbs_config["next_batch_selector_params"]
    batch_sizes = next_batch_selector_params.pop("batch_size", None)
    # sample random param 
    param_grid = SimulationParameterGrid(next_batch_selector_params)
    np.random.seed(rnd_seed)
    param_idx = np.random.randint(len(param_grid), size=1, dtype='int64')[0]
    next_batch_selector_params = param_grid[param_idx]
    next_batch_selector_params["batch_size"] = batch_sizes
    return next_batch_selector_params
    
"""
    Random sample from the given parameter set using the 
    distribution given in the config file.
    If use_uniform=True, then samples each parameter uniformly.
"""
def get_param_from_dist(nbs_config,
                        rnd_seed=0,
                        use_uniform=False,
                        exploration_strategy='weighted'):
    nbs_params = nbs_config["next_batch_selector_params"]
    nbs_params_probas = nbs_config["nbs_params_probas"]
    # sample random param 
    np.random.seed(rnd_seed)
    sorted_params = sorted(nbs_params_probas.keys())
    
    if exploration_strategy not in nbs_params["exploration_strategy"]:
        raise ValueError('Given exploration strategy not supported in config file.')
    
    nbs_params["exploration_strategy"] = exploration_strategy
    if exploration_strategy == 'random' or exploration_strategy == 'dissimilar':
        for removable_param, default_value in [('exploration_use_quantile_for_weight', False), 
                                               ('exploration_weight_threshold', 0.0), 
                                               ('exploration_beta', 0.0), 
                                               ('exploration_dissimilarity_lambda', 0.0)]:
            nbs_params[removable_param] = default_value
            sorted_params.remove(removable_param)
                
    while len(sorted_params) > 0:
        param = sorted_params.pop()
        param_choices = np.array(nbs_params[param])
        param_probas = nbs_params_probas[param]
        if param_choices.ndim > 1:
            param_choices = param_choices.flatten()
            
        if use_uniform:
            param_probas = [1.0/len(param_choices) for _ in range(len(param_choices))] # discrete uniform sampling
        param_sampled_choice = np.random.choice(param_choices, size=1, p=param_probas)[0]
        
        # modify nbs_params dict with sampled choice
        nbs_params[param] = param_sampled_choice
        
    nbs_params["class"] = nbs_params["class"][0]
    return nbs_params
	
	
"""
    Evaluates selected batch by assuming all are active/hits.
"""
def evaluate_selected_batch(exploitation_df, exploration_df, 
							exploitation_array, exploration_array,
							params_set_results_dir,
							pipeline_config,
							iter_num,
                            batch_size,
                            total_selection_time,
                            add_mean_medians=False):
    w_novelty = pipeline_config['common']['metrics_params']['w_novelty']
    perc_vec = pipeline_config['common']['metrics_params']['perc_vec']
    task_names = pipeline_config['common']['task_names']
    cost_col_name = pipeline_config['unlabeled_data_params']['cost_col_name']
    iter_results_dir = params_set_results_dir+'/'+pipeline_config['common']['iter_results_dir'].format(iter_num)
    eval_dest_file = iter_results_dir+'/'+pipeline_config['common']['eval_dest_file']
    pathlib.Path(eval_dest_file).parent.mkdir(parents=True, exist_ok=True)

    cols_names = task_names
    if add_mean_medians:
        cols_names = cols_names+['Mean', 'Median']
    # retrieve max_hits_list, max_cluster_hits_list of the unlabeled data for this iteration
    max_hits_list, max_cluster_hits_list, max_novel_hits_list = get_unlabeled_maxes(training_loader_params=pipeline_config['training_data_params'], 
                                                                                    unlabeled_loader_params=pipeline_config['unlabeled_data_params'],
                                                                                    task_names=task_names,
                                                                                    batch_size=batch_size)
    train_clusters = prepare_loader(data_loader_params=pipeline_config['training_data_params'],
                                    task_names=task_names).get_clusters()
    
    exploitation_batch_size, exploitation_batch_cost = 0, 0
    if exploitation_df is not None:
        exploitation_df.to_csv(iter_results_dir+'/'+pipeline_config['common']['batch_csv'].format('exploitation'),
                               index=False)
        exploitation_metrics_mat, metrics_names = eval_on_metrics(exploitation_df[task_names].values, np.ones_like(exploitation_df[task_names].values), 
                                                                  train_clusters, exploitation_array[:,1],
                                                                  max_hits_list, max_cluster_hits_list, max_novel_hits_list,
                                                                  add_mean_medians, w_novelty, perc_vec)
        exploitation_batch_size = exploitation_df[task_names].shape[0]
        try:
            exploitation_costs = exploitation_df[cost_col_name].values.astype(float)
        except:
            exploitation_costs = np.ones(shape=(exploitation_df.shape[0],))
        exploitation_batch_cost = np.sum(exploitation_costs)
    else:
        exploitation_metrics_mat, metrics_names = eval_on_metrics(None, None, 
                                                                  train_clusters, None,
                                                                  max_hits_list, max_cluster_hits_list, max_novel_hits_list,
                                                                  add_mean_medians, w_novelty, perc_vec)
    exploration_batch_size, exploration_batch_cost = 0, 0
    if exploration_df is not None:
        exploration_df.to_csv(iter_results_dir+'/'+pipeline_config['common']['batch_csv'].format('exploration'),
                              index=False)
        exploration_metrics_mat, metrics_names = eval_on_metrics(exploration_df[task_names].values, np.ones_like(exploration_df[task_names].values), 
                                                                 train_clusters, exploration_array[:,1],
                                                                 max_hits_list, max_cluster_hits_list, max_novel_hits_list,
                                                                 add_mean_medians, w_novelty, perc_vec)
        exploration_batch_size = exploration_df[task_names].shape[0]
        try:
            exploration_costs = exploration_df[cost_col_name].values.astype(float)
        except:
            exploration_costs = np.ones(shape=(exploration_df.shape[0],))
        exploration_batch_cost = np.sum(exploration_costs)
    else:
        exploration_metrics_mat, metrics_names = eval_on_metrics(None, None,
                                                                 train_clusters, None,
                                                                 max_hits_list, max_cluster_hits_list, max_novel_hits_list,
                                                                 add_mean_medians, w_novelty, perc_vec)
    # record rest of metrics
    exploitation_metrics_mat = np.vstack([exploitation_metrics_mat, [[exploitation_batch_size], [exploitation_batch_cost]]])
    exploration_metrics_mat = np.vstack([exploration_metrics_mat, [[exploration_batch_size], [exploration_batch_cost]]])
    
    # construct exploitation + exploration metrics
    total_df = pd.concat([exploitation_df, exploration_df])
    if (exploitation_df is not None) and (exploration_df is not None):
        total_array = np.vstack([exploitation_array, exploration_array])
    elif  (exploitation_df is not None) and (exploration_df is None):
        total_array = exploitation_array
    elif  (exploitation_df is None) and (exploration_df is not None):
        total_array = exploration_array
    else:
        raise ValueError('Error in evaluating batch: total selection array is empty.')
        
    total_metrics_mat, metrics_names = eval_on_metrics(total_df[task_names].values, np.ones_like(total_df[task_names].values), 
                                                       train_clusters, total_array[:,1],
                                                       max_hits_list, max_cluster_hits_list, max_novel_hits_list,
                                                       add_mean_medians, w_novelty, perc_vec)
    metrics_names = metrics_names + ['batch_size', 'batch_cost']
    
    total_batch_size = exploitation_batch_size + exploration_batch_size
    try:
        total_batch_cost = total_df[cost_col_name].values.astype(float)
    except:
        total_batch_cost = np.ones(shape=(total_df.shape[0],))
    total_batch_cost = np.sum(total_batch_cost)
    total_metrics_mat = np.vstack([total_metrics_mat, [[total_batch_size], [total_batch_cost]]])
    
    total_cherry_picking_time = total_batch_size * pipeline_config['common']['cherry_picking_time_per_cpd']
    screening_time_per_batch = pipeline_config['common']['screening_time_per_batch'] 
    total_screening_time = total_cherry_picking_time + screening_time_per_batch
    
    metrics_mat = np.vstack([exploitation_metrics_mat, exploration_metrics_mat, total_metrics_mat, 
                            [[total_cherry_picking_time]], [[screening_time_per_batch]], [[total_screening_time]]])
    metrics_names = ['exploitation_'+m for m in metrics_names] + \
                    ['exploration_'+m for m in metrics_names] + \
                    ['total_'+m for m in metrics_names] + \
                    ['total_cherry_picking_time', 'screening_time_per_batch', 'total_screening_time']
                    
    # save to destination
    metrics_df = pd.DataFrame(data=metrics_mat,
                              columns=[iter_num],
                              index=metrics_names).T
    metrics_df.index.name = 'iter_num'
    metrics_df.to_csv(eval_dest_file, index=True)
	
"""
    Summarize simulation evaluation results by aggregating.
"""
def summarize_simulation(params_set_results_dir,
						 pipeline_config):
    summary_dest_file = params_set_results_dir+'/'+pipeline_config['common']['summary_dest_file']
    pathlib.Path(summary_dest_file).parent.mkdir(parents=True, exist_ok=True)

    metrics_df_list = []
    iter_dirs = glob.glob(params_set_results_dir+'/*/')
    for i in range(len(iter_dirs)):
        iter_d = params_set_results_dir+'/'+pipeline_config['common']['iter_results_dir'].format(i)
        eval_dest_file = iter_d+'/'+pipeline_config['common']['eval_dest_file']
        if not os.path.exists(eval_dest_file):
            print(eval_dest_file, '\nDoes not exist.')
        else:
            metrics_df_list.append(pd.read_csv(eval_dest_file))

    metrics_df_concat = pd.concat(metrics_df_list)
    metrics_ordering = [m for m in metrics_df_concat.columns if 'ratio' not in m or 'exploration' in m] + [m for m in metrics_df_concat.columns if 'ratio' in m and 'exploration' not in m]
    summary_df = pd.concat([metrics_df_concat[[m for m in metrics_df_concat.columns if 'ratio' not in m or 'exploration' in m]].sum(),
                            metrics_df_concat[[m for m in metrics_df_concat.columns if 'ratio' in m and 'exploration' not in m]].mean()]).to_frame().T
    summary_df.iloc[-1,0] = 9999
    summary_df = pd.concat([metrics_df_concat[metrics_ordering], summary_df])
    summary_df.to_csv(summary_dest_file, index=False)
    
class SimulationParameterGrid(ParameterGrid):
    """
    Custom parameter grid class due to sklearn's ParameterGrid restriction to int32.
    """

    def __getitem__(self, ind):
        """
        Same as sklearn's ParameterGrid class but np.product(sizes, dtype='int64').
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes, dtype='int64')
            
            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('SimulationParameterGrid index out of range')