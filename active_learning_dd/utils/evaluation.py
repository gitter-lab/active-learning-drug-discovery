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
def _get_metrics_names(perc_vec):
    return ['hits'] + \
           ['max_hits'] + \
           ['norm_hits_ratio'] + \
           ['cluster_hits'] + \
           ['max_cluster_hits'] + \
           ['norm_cluster_hits_ratio'] + \
           ['novel_hits'] + \
           ['max_novel_hits'] + \
           ['norm_novel_hits_ratio'] + \
           ['ROC AUC ratio', 'PR AUC ratio'] + \
           ['NEF {}% ratio'.format(perc*100) for perc in perc_vec] + \
           ['Random NEF {}% ratio'.format(perc*100) for perc in perc_vec] + \
           ['NEF AUC ratio', 'Random NEF AUC ratio'] + \
           ['EF {}% ratio'.format(perc*100) for perc in perc_vec] + \
           ['MAX EF {}% ratio'.format(perc*100) for perc in perc_vec]

"""
    Helper function that evaluates selected batch on metrics.
"""
def eval_on_metrics(y_true, y_preds,
                    train_clusters, test_clusters,
                    max_hits_list, max_cluster_hits_list, max_novel_hits_list,
                    add_mean_medians, w_novelty, perc_vec):
    if y_true is None:
        metrics_names = _get_metrics_names(perc_vec)
        metrics_res_list = []  
        metrics_res_mat = np.ones(shape=(len(metrics_names),1))*np.nan
        return metrics_res_mat, metrics_names
        
    max_hits_list = [min(x, y_true.shape[0]) for x in max_hits_list]
    max_cluster_hits_list = [min(x, y_true.shape[0]) for x in max_cluster_hits_list]
    max_novel_hits_list = [min(x, y_true.shape[0]) for x in max_novel_hits_list]
    n_tests_list = [y_true.shape[0]]
    metrics_names = _get_metrics_names(perc_vec)
    
    # process roc and pr auc
    roc_auc_arr = roc_auc(y_true, y_preds)
    pr_auc_arr = pr_auc(y_true, y_preds)
    
    # process ef-based metrics
    nef_auc_arr, random_nef_auc_arr, nef_mat, random_nef_mat, ef_mat, max_ef_mat = nef_auc(y_true, y_preds, perc_vec)
    
    # process n_hits and n_cluster_hits based metrics
    _, norm_hits_ratio_mat, n_hits_mat, max_n_hits_mat, norm_cluster_hits_ratio_mat, n_cluster_hits_mat, max_n_cluster_hits_mat = novel_n_hits(y_true, y_preds, test_clusters, 
                                                                                                                                               n_tests_list, w_novelty)
                                                                                                                                               
    novel_cluster_hits_mat, max_novel_cluster_hits_mat, norm_novel_cluster_hits_ratio_mat = novel_cluster_n_hits(y_true, y_preds, [train_clusters, test_clusters], n_tests_list)             
    # modify the maxes and ratios
    num_tasks = 1
    if y_true.ndim > 1:
        num_tasks = y_true.shape[-1]
    max_n_hits_mat = np.array(max_hits_list).reshape(-1,num_tasks)
    max_n_cluster_hits_mat = np.array(max_cluster_hits_list).reshape(-1,num_tasks)
    max_novel_cluster_hits_mat = np.array(max_novel_hits_list).reshape(-1,num_tasks)
    norm_hits_ratio_mat = n_hits_mat / max_n_hits_mat
    norm_cluster_hits_ratio_mat = n_cluster_hits_mat / max_n_cluster_hits_mat
    norm_novel_cluster_hits_ratio_mat = novel_cluster_hits_mat / max_novel_cluster_hits_mat

    # append to list with desired ordering
    metrics_res_list = []  
    metrics_res_list.append(n_hits_mat)
    metrics_res_list.append(max_n_hits_mat)
    metrics_res_list.append(norm_hits_ratio_mat)

    metrics_res_list.append(n_cluster_hits_mat)
    metrics_res_list.append(max_n_cluster_hits_mat)
    metrics_res_list.append(norm_cluster_hits_ratio_mat)

    metrics_res_list.append(novel_cluster_hits_mat)
    metrics_res_list.append(max_novel_cluster_hits_mat)
    metrics_res_list.append(norm_novel_cluster_hits_ratio_mat)
    
    metrics_res_list.append(roc_auc_arr)
    metrics_res_list.append(pr_auc_arr)
    
    metrics_res_list.append(nef_mat)
    metrics_res_list.append(random_nef_mat)
    metrics_res_list.append(nef_auc_arr)
    metrics_res_list.append(random_nef_auc_arr)
    metrics_res_list.append(ef_mat)
    metrics_res_list.append(max_ef_mat)
    
    # add mean and median columns
    metrics_res_mat = np.vstack(metrics_res_list)
    if add_mean_medians:
        mean_arr = np.mean(metrics_res_mat, axis=1).reshape(-1,1)
        median_arr = np.median(metrics_res_mat, axis=1).reshape(-1,1)
        metrics_res_mat = np.hstack([metrics_res_mat, mean_arr, median_arr])

    return metrics_res_mat, metrics_names