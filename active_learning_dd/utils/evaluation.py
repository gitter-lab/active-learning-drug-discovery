"""
    Contains evaluation functions. Adapted from: https://github.com/gitter-lab/pria_lifechem/blob/master/pria_lifechem/evaluation.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import pathlib

from .metrics import *

"""
    Helper function that maintains the list of metric names.
"""
def _get_metrics_names(n_tests_list):
    return ['n_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['max_n_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['n_cluster_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['max_n_cluster_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['norm_hits_ratio_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['norm_cluster_hits_ratio_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['novel_n_hits_at_{}'.format(n_tests) for n_tests in n_tests_list]

"""
    Helper function that evaluates selected batch on metrics.
"""
def _eval_on_metrics(y_true, y_preds, clusters,
                     max_hits_list, max_cluster_hits_list,
                     add_mean_medians, w_novelty):
    max_hits_list = [min(x, y_true.shape[0]) for x in max_hits_list]
    max_cluster_hits_list = [min(x, y_true.shape[0]) for x in max_cluster_hits_list]
    n_tests_list = [y_true.shape[0]]
    metrics_names = _get_metrics_names(n_tests_list)
   
    # process n_hits and n_cluster_hits based metrics
    novel_n_hits_mat, norm_hits_ratio_mat, n_hits_mat, max_n_hits_mat, norm_cluster_hits_ratio_mat, n_cluster_hits_mat, max_n_cluster_hits_mat = novel_n_hits(y_true, y_preds, clusters, 
                                                                                                                                                              n_tests_list, w_novelty)
    # modify the maxes and ratios
    max_n_hits_mat = np.array(max_hits_list).reshape(-1,y_true.shape[1])
    max_n_cluster_hits_mat = np.array(max_cluster_hits_list).reshape(-1,y_true.shape[1])
    norm_hits_ratio_mat = n_hits_mat / max_n_hits_mat
    norm_cluster_hits_ratio_mat = n_cluster_hits_mat / max_n_cluster_hits_mat
    
    # append to list with desired ordering
    metrics_res_list = []  
    metrics_res_list.append(n_hits_mat)
    metrics_res_list.append(max_n_hits_mat)
    metrics_res_list.append(n_cluster_hits_mat)
    metrics_res_list.append(max_n_cluster_hits_mat)
    metrics_res_list.append(norm_hits_ratio_mat)
    metrics_res_list.append(norm_cluster_hits_ratio_mat)
    metrics_res_list.append(novel_n_hits_mat)
    
    # add mean and median columns
    metrics_res_mat = np.vstack(metrics_res_list)
    if add_mean_medians:
        mean_arr = np.mean(metrics_res_mat, axis=1).reshape(-1,1)
        median_arr = np.median(metrics_res_mat, axis=1).reshape(-1,1)
        metrics_res_mat = np.hstack([metrics_res_mat, mean_arr, median_arr])
    
    return metrics_res_mat, metrics_names    
    
    
"""
    Evaluates selected batch by assuming all are active/hits.
"""
def evaluate_selected_batch(y_true, clusters,
                            max_hits_list, max_cluster_hits_list,
                            task_names, eval_dest_file,
                            add_mean_medians=True, w_novelty=0.5):
    # create directories in case they don't exist
    pathlib.Path(eval_dest_file).parent.mkdir(parents=True, exist_ok=True)
    
    # evaluate on metrics
    y_preds = np.ones_like(y_true) # assume all preds are true
    metrics_mat, metrics_names = _eval_on_metrics(y_true, y_preds, clusters,
                                                  max_hits_list, max_cluster_hits_list,
                                                  add_mean_medians, w_novelty)
    
    # construct pd dataframe
    cols_names = task_names
    if add_mean_medians:
        cols_names = cols_names+['Mean', 'Median']
        
    metrics_df = pd.DataFrame(data=metrics_mat,
                              columns=cols_names,
                              index=metrics_names)

    # save to destination
    metrics_df.to_csv(eval_dest_file, index=True)