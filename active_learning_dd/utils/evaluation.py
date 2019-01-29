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
    return ['hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['max_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['norm_hits_ratio_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['cluster_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['max_cluster_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['norm_cluster_hits_ratio_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['novel_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['max_novel_hits_at_{}'.format(n_tests) for n_tests in n_tests_list] + \
           ['norm_novel_hits_ratio_at_{}'.format(n_tests) for n_tests in n_tests_list]

"""
    Helper function that evaluates selected batch on metrics.
"""
def eval_on_metrics(y_true, y_preds,
                    train_clusters, test_clusters,
                    max_hits_list, max_cluster_hits_list, max_novel_hits_list,
                    add_mean_medians, w_novelty):
    max_hits_list = [min(x, y_true.shape[0]) for x in max_hits_list]
    max_cluster_hits_list = [min(x, y_true.shape[0]) for x in max_cluster_hits_list]
    max_novel_hits_list = [min(x, y_true.shape[0]) for x in max_novel_hits_list]
    n_tests_list = [y_true.shape[0]]
    metrics_names = _get_metrics_names(n_tests_list)

    # process n_hits and n_cluster_hits based metrics
    _, norm_hits_ratio_mat, n_hits_mat, max_n_hits_mat, norm_cluster_hits_ratio_mat, n_cluster_hits_mat, max_n_cluster_hits_mat = novel_n_hits(y_true, y_preds, test_clusters, 
                                                                                                                                               n_tests_list, w_novelty)
                                                                                                                                               
    novel_cluster_hits_mat, max_novel_cluster_hits_mat, norm_novel_cluster_hits_ratio_mat = novel_cluster_n_hits(y_true, y_pred, [train_clusters, test_clusters], n_tests_list)             
    # modify the maxes and ratios
    max_n_hits_mat = np.array(max_hits_list).reshape(-1,y_true.shape[1])
    max_n_cluster_hits_mat = np.array(max_cluster_hits_list).reshape(-1,y_true.shape[1])
    max_novel_cluster_hits_mat = np.array(max_novel_hits_list).reshape(-1,y_true.shape[1])
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

    # add mean and median columns
    metrics_res_mat = np.vstack(metrics_res_list)
    if add_mean_medians:
        mean_arr = np.mean(metrics_res_mat, axis=1).reshape(-1,1)
        median_arr = np.median(metrics_res_mat, axis=1).reshape(-1,1)
        metrics_res_mat = np.hstack([metrics_res_mat, mean_arr, median_arr])

    return metrics_res_mat, metrics_names
    
    
