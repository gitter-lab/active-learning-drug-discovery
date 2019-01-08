"""
    Contains metrics functions. Adapted from: https://github.com/gitter-lab/pria_lifechem/blob/master/pria_lifechem/evaluation.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score


"""
    Helper function for metrics for a single or multiple tasks.
"""
def _metric_calc(y_true, y_pred, metric_func):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    num_tasks = y_true.shape[1]
    metric_res = np.zeros(num_tasks)
    for ti in range(num_tasks):
        y_true_ti = y_true[:, ti]
        y_pred_ti = y_pred[:, ti]
        
        # remove missing labels
        non_missing_labels_ti = np.where(~np.isnan(y_true_ti))[0] 
        y_true_ti = y_true_ti[non_missing_labels_ti]
        y_pred_ti = y_pred_ti[non_missing_labels_ti]
        
        try:
            metric_res[ti] = metric_func(y_true_ti, y_pred_ti)
        except ValueError:
            metric_res[ti] = np.nan
            
    return metric_res
    
"""
    Helper function for metrics that loop on extra list parameters. 
"""
def _metric_with_param_list(y_true, y_pred, param_list, metric_func):
    metric_list = []
    for curr_param in param_list:
        curr_metric_calc = metric_func(y_true, y_pred, curr_param)
        metric_list.append(curr_metric_calc)
    return np.vstack(metric_list)

    
"""
    Computes ROC AUC for a single or multiple tasks.
"""
def roc_auc(y_true, y_pred):
    return _metric_calc(y_true, y_pred, roc_auc_score)
    
    
"""
    Computes PR AUC for a single or multiple tasks.
"""
def pr_auc(y_true, y_pred):
    return _metric_calc(y_true, y_pred, average_precision_score)


"""
    Computes enrichment factor vector at the given percentile for multiple tasks.
"""
def enrichment_factor_single_perc(y_true, y_pred, percentile):
    def ef_calc(y_true_ti, y_pred_ti):
        sample_size = int(y_true_ti.shape[0] * percentile)
        indices = np.argsort(y_pred_ti, axis=0)[::-1][:sample_size]
        
        n_actives = np.nansum(y_true_ti) 
        n_experimental = np.nansum( y_true_ti[indices] )
        return ( float(n_experimental) /  n_actives ) / percentile 
        
    return _metric_calc(y_true, y_pred, ef_calc)

    
"""
    Computes max enrichment factor vector at the given percentile for multiple tasks.
"""
def max_enrichment_factor_single_perc(y_true, y_pred, percentile):
    def max_ef_calc(y_true_ti, y_pred_ti):
        n_actives = np.nansum(y_true_ti) 
        sample_size = int(y_true_ti.shape[0] * percentile)
        return ( min(n_actives, sample_size) /  n_actives ) / percentile 
        
    return _metric_calc(y_true, y_pred, max_ef_calc)
    
    
"""
    Computes enrichment factor vector at the percentile vectors. 
"""
def enrichment_factor(y_true, y_pred, perc_vec):
    return _metric_with_param_list(y_true, y_pred, perc_vec, enrichment_factor_single_perc)


"""
    Computes max enrichment factor vector at the percentile vectors.
"""   
def max_enrichment_factor(y_true, y_pred, perc_vec): 
    return _metric_with_param_list(y_true, y_pred, perc_vec, max_enrichment_factor_single_perc)
    

"""
    Computes normalized enrichment factor vector at the percentile vectors. 
"""       
def norm_enrichment_factor(y_true, y_pred, perc_vec): 
    ef_mat = enrichment_factor(y_true, y_pred, perc_vec)
    max_ef_mat = max_enrichment_factor(y_true, y_pred, perc_vec)
    nef_mat = ef_mat / max_ef_mat
    random_nef_mat = 1 / max_ef_mat
    return nef_mat, random_nef_mat, ef_mat, max_ef_mat


"""
    Computes the NEF AUC and Random NEF AUC at the percentile vectors.
"""
def nef_auc(y_true, y_pred, perc_vec):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    num_tasks = y_true.shape[1]
    nef_mat, random_nef_mat, ef_mat, max_ef_mat = norm_enrichment_factor(y_true, 
                                                                         y_pred, 
                                                                         perc_vec)
    nef_auc_arr = np.zeros(num_tasks)
    random_nef_auc_arr = np.zeros(num_tasks) 
    for ti in range(num_tasks):
        nef_auc_arr[ti] = auc(perc_vec, nef_mat[:,ti])
        random_nef_auc_arr[ti] = auc(perc_vec, random_nef_mat[:,ti])
        
    return nef_auc_arr, random_nef_auc_arr, nef_mat, random_nef_mat, ef_mat, max_ef_mat


"""
    Computes number of actives found in the top n_tests in n_tests_list.
"""   
def n_hits_calc(y_true, y_pred, n_tests_list):
    return _metric_with_param_list(y_true, y_pred, n_tests_list, n_hits_calc_at_n_tests)

 
"""
    Computes number of actives found in the top n_tests.
"""   
def n_hits_calc_at_n_tests(y_true, y_pred, n_tests):
    def n_hits(y_true_ti, y_pred_ti):
        indices = np.argsort(y_pred_ti, axis=0)[::-1][:n_tests]
        return np.nansum( y_true_ti[indices] )
        
    return _metric_calc(y_true, y_pred, n_hits)


"""
    Computes max number of actives found in the top n_tests in n_tests_list.
"""   
def max_n_hits_calc(y_true, y_pred, n_tests_list):
    return _metric_with_param_list(y_true, y_pred, n_tests_list, max_n_hits_calc_at_n_tests)

 
"""
    Computes max number of actives found in the top n_tests.
"""   
def max_n_hits_calc_at_n_tests(y_true, y_pred, n_tests):
    def max_n_hits(y_true_ti, y_pred_ti):
        indices = np.argsort(y_true_ti, axis=0)[::-1][:n_tests]
        return np.nansum( y_true_ti[indices] )
        
    return _metric_calc(y_true, y_pred, max_n_hits)


"""
    Helper function for metrics for a single or multiple tasks and cluster data.
"""
def _metric_calc_with_clusters(y_true, y_pred, clusters, metric_func):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if clusters.ndim == 1:
        clusters = clusters.reshape(-1, 1)
    
    num_tasks = y_true.shape[1]
    metric_res = np.zeros(num_tasks)
    for ti in range(num_tasks):
        y_true_ti = y_true[:, ti]
        y_pred_ti = y_pred[:, ti]
        
        # remove missing labels
        non_missing_labels_ti = np.where(~np.isnan(y_true_ti))[0] 
        y_true_ti = y_true_ti[non_missing_labels_ti]
        y_pred_ti = y_pred_ti[non_missing_labels_ti]
        clusters_ti = clusters[non_missing_labels_ti]
        
        try:
            metric_res[ti] = metric_func(y_true_ti, y_pred_ti, clusters_ti)
        except ValueError:
            metric_res[ti] = np.nan
            
    return metric_res
    
"""
    Helper function for metrics that loop on extra list parameters and cluster data. 
"""
def _metric_with_clusters_and_param_list(y_true, y_pred, clusters, 
                                         param_list, metric_func):
    metric_list = []
    for curr_param in param_list:
        curr_metric_calc = metric_func(y_true, y_pred, clusters, curr_param)
        metric_list.append(curr_metric_calc)
    return np.vstack(metric_list)
    
    
"""
    Computes number of clusters with actives found in the top n_tests in n_tests_list.
"""   
def n_cluster_hits_calc(y_true, y_pred, clusters, n_tests_list):
    return _metric_with_clusters_and_param_list(y_true, y_pred, clusters, 
                                                n_tests_list, 
                                                n_cluster_hits_calc_at_n_tests)

 
"""
    Computes number of clusters with actives found in the top n_tests.
"""   
def n_cluster_hits_calc_at_n_tests(y_true, y_pred, clusters, n_tests):
    def n_cluster_hits(y_true_ti, y_pred_ti, clusters_ti):
        indices = np.argsort(y_pred_ti, axis=0)[::-1][:n_tests]
        y_true_ti = y_true_ti[indices]
        clusters_ti = clusters_ti[indices]
        
        # Get the clusters with actives
        active_indices = np.where(y_true_ti == 1)[0]
        clusters_with_actives_ti = clusters_ti[active_indices]
        num_clusters_with_actives_ti = np.unique(clusters_with_actives_ti).shape[0]
        return num_clusters_with_actives_ti
        
    return _metric_calc_with_clusters(y_true, y_pred, clusters, n_cluster_hits)


"""
    Computes max number of clusters with actives found in the top n_tests in n_tests_list.
"""   
def max_n_cluster_hits_calc(y_true, y_pred, clusters, n_tests_list):
    return _metric_with_clusters_and_param_list(y_true, y_pred, clusters,
                                                n_tests_list, 
                                                max_n_cluster_hits_calc_at_n_tests)

 
"""
    Computes max number of clusetrs with actives found in the top n_tests.
"""   
def max_n_cluster_hits_calc_at_n_tests(y_true, y_pred, clusters, n_tests):
    def max_n_cluster_hits(y_true_ti, y_pred_ti, clusters_ti):
        # Get the clusters with actives
        active_indices = np.where(y_true_ti == 1)[0]
        clusters_with_actives_ti = clusters_ti[active_indices]
        max_clusters_with_actives_ti = min(active_indices.shape[0], 
                                           np.unique(clusters_with_actives_ti).shape[0])
        return max_clusters_with_actives_ti
    return _metric_calc_with_clusters(y_true, y_pred, clusters, max_n_cluster_hits)

    
"""
    Computes normalized hits ratio at the n_tests in n_tests_list. 
"""       
def norm_hits_ratio(y_true, y_pred, n_tests_list): 
    n_hits_mat = n_hits_calc(y_true, y_pred, n_tests_list)
    max_n_hits_mat = max_n_hits_calc(y_true, y_pred, n_tests_list) 
    norm_hits_ratio_mat = n_hits_mat / max_n_hits_mat
    return norm_hits_ratio_mat, n_hits_mat, max_n_hits_mat
    
    
"""
    Computes normalized cluster hits ratio at the n_tests in n_tests_list. 
"""       
def norm_cluster_hits_ratio(y_true, y_pred, clusters, n_tests_list): 
    n_cluster_hits_mat = n_cluster_hits_calc(y_true, y_pred, clusters, n_tests_list)
    max_n_cluster_hits_mat = max_n_cluster_hits_calc(y_true, y_pred, clusters, n_tests_list) 
    norm_cluster_hits_ratio_mat = n_cluster_hits_mat / max_n_cluster_hits_mat
    return norm_cluster_hits_ratio_mat, n_cluster_hits_mat, max_n_cluster_hits_mat
    

"""
    Computes novelty hits at the n_tests in n_tests_list. 
"""       
def novel_n_hits(y_true, y_pred, clusters, n_tests_list, w=0.5): 
    norm_hits_ratio_mat, n_hits_mat, max_n_hits_mat = norm_hits_ratio(y_true, y_pred, n_tests_list)
    norm_cluster_hits_ratio_mat, n_cluster_hits_mat, max_n_cluster_hits_mat = norm_cluster_hits_ratio(y_true, y_pred, clusters, n_tests_list) 
    novel_n_hits_mat = (w * norm_hits_ratio_mat) + ((1-w) * norm_cluster_hits_ratio_mat)
    return novel_n_hits_mat, norm_hits_ratio_mat, n_hits_mat, max_n_hits_mat, norm_cluster_hits_ratio_mat, n_cluster_hits_mat, max_n_cluster_hits_mat
    