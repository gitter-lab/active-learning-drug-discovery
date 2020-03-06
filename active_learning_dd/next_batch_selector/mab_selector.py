"""
    Contains Mult-Arm Bandit (MAB) selectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nbs_base import NBSBase
from ..utils.data_utils import get_avg_cluster_dissimilarity, get_avg_cluster_dissimilarity_from_file
import pandas as pd
import numpy as np
import time

class MABSelector(NBSBase):
    """
    MAB class for next batch selectors. Each cluster is an arm. 
    Adapted from similar idea to: https://link.springer.com/article/10.1007/s41109-019-0145-0
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 batch_size=384,
                 uncertainty_method="least_confidence",
                 uncertainty_alpha=0.5):
        super(MABSelector, self).__init__(training_loader=training_loader,
                                          unlabeled_loader=unlabeled_loader,
                                          trained_model=trained_model,
                                          batch_size=batch_size)
        self.uncertainty_method = uncertainty_method
        self.uncertainty_params_list = None
        if isinstance(uncertainty_method, list):
            self.uncertainty_method = uncertainty_method[0]
            if len(uncertainty_method) > 1:
                self.uncertainty_params_list = [self.feature_dist_func]
                self.uncertainty_params_list += uncertainty_method[1:]
        
        self.uncertainty_alpha = uncertainty_alpha
        self.clusters_unlabeled = self.unlabeled_loader.get_clusters()
        
        # create pandas df for various cluster calculations
        u_clusters, c_clusters = np.unique(self.clusters_unlabeled,
                                           return_counts=True)
        self.total_clusters = len(u_clusters)
        self.cluster_cols = ['Cluster ID', 'Cluster Mol Count', 
                             'Mean Uncertainty',
                             'Mean Activity Prediction']
        self.clusters_df = pd.DataFrame(data=np.nan*np.zeros((self.total_clusters, len(self.cluster_cols))),
                                        columns=self.cluster_cols)
        self.clusters_df['Cluster ID'] = u_clusters
        self.clusters_df['Cluster Mol Count'] = c_clusters
        self.clusters_df.index = self.clusters_df['Cluster ID']
            
    def _compute_cluster_activity_prediction(self):
        preds_unlabeled = self.trained_model.predict(self.unlabeled_loader.get_features())[:,0] # get first task for now. TODO: account for multi-task setting?
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            cluster_preds = preds_unlabeled[mol_idx]
            avg_cluster_activity_i = np.nan_to_num(np.mean(cluster_preds))
            self.clusters_df.loc[ci, 'Mean Activity Prediction'] = avg_cluster_activity_i
    
    def _compute_cluster_uncertainty(self):
        uncertainty_unlabeled = self.trained_model.get_uncertainty(X=self.unlabeled_loader.get_features(), 
                                                                   uncertainty_method=self.uncertainty_method,
                                                                   uncertainty_params_list=self.uncertainty_params_list)
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            cluster_uncertainty = uncertainty_unlabeled[mol_idx]
            avg_cluster_uncertainty_i = np.nan_to_num(np.mean(cluster_uncertainty))
            self.clusters_df.loc[ci, 'Mean Uncertainty'] = avg_cluster_uncertainty_i
    
    """
        Also selects instances from clusters based on UCB.
    """
    def _get_max_ucb_instance_from_cluster(self, cluster_cid, prev_selected_instances_from_cluster):
        instances_idx = np.where(self.clusters_unlabeled == cluster_cid)[0]
        instances_idx = np.setdiff1d(instances_idx, prev_selected_instances_from_cluster)
        X_instances = self.unlabeled_loader.get_features()[instances_idx]
        instances_preds = self.trained_model.predict(X_instances)[:,0]
        instances_uncertainty = self.trained_model.get_uncertainty(X=X_instances, 
                                                                   uncertainty_method=self.uncertainty_method,
                                                                   uncertainty_params_list=self.uncertainty_params_list)[:,0]
        instances_ucb = instances_preds + (self.uncertainty_alpha * instances_uncertainty)
        max_ucb_instance_argmax = np.argmax(instances_ucb)
        max_ucb_instance_idx = instances_idx[max_ucb_instance_argmax]
        
        # update cluster information
        # TODO: maybe bad design to put updating in this function; becuase it is unrelated to selecting max ucb instance.
        remaining_instances = np.setdiff1d(np.arange(len(instances_idx)), [max_ucb_instance_argmax])
        if len(remaining_instances) > 0:
            avg_cluster_activity = np.nan_to_num(np.mean(instances_preds[remaining_instances]))
            self.clusters_df.loc[cluster_cid, 'Mean Activity Prediction'] = avg_cluster_activity
            avg_cluster_uncertainty = np.nan_to_num(np.mean(instances_uncertainty[remaining_instances]))
            self.clusters_df.loc[cluster_cid, 'Mean Uncertainty'] = avg_cluster_uncertainty
            self.clusters_df.loc[cluster_cid, 'Cluster Mol Count'] = len(remaining_instances)
        else: # drop cluster if no more instances to select
            self.clusters_df = self.clusters_df.drop(cluster_cid, axis=0)
        return max_ucb_instance_idx
            
    """
        Uses UCB with mean of cluster/arm estimated by average model prediction. 
        Uncertainty measured by model uncertainty.
    """
    def select_next_batch(self):
        current_budget = self.batch_size
        selected_cluster_instances_dict = {}
        # populate self.clusters_df
        self._compute_cluster_uncertainty()
        self._compute_cluster_activity_prediction()
        
        # select max ucb cluster, then select max ucb instance from that cluster, 
        # recalculate ucb after removing selected instance, then repeat until entire budget exhausted.
        # TODO: optimize recalculation of argmax since only means for the previously selected cluster is updated each iteration.
        while current_budget > 0:
            ucb_uncertainty = self.uncertainty_alpha * self.clusters_df['Mean Uncertainty']
            ucb_clusters = self.clusters_df['Mean Activity Prediction'] + ucb_uncertainty
            ucb_max_cid = self.clusters_df['Cluster ID'].iloc[np.argmax(ucb_clusters.values)]
            
            if ucb_max_cid not in selected_cluster_instances_dict:
                selected_cluster_instances_dict[ucb_max_cid] = []
            
            selected_instance_idx = self._get_max_ucb_instance_from_cluster(ucb_max_cid, selected_cluster_instances_dict[ucb_max_cid])
                    
            selected_cluster_instances_dict[ucb_max_cid].append(selected_instance_idx)
            current_budget -= 1
            
        selected_cluster_instances_pairs = [(k, selected_cluster_instances_dict[k]) for k in selected_cluster_instances_dict]
        return (selected_cluster_instances_pairs, []) # exploration selection is None in this case
        
        
class MABUncertaintySelector(MABSelector):
    """
    MAB class that simply selects most uncertain clusters.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 batch_size=384,
                 uncertainty_method="least_confidence"):
        super(MABUncertaintySelector, self).__init__(training_loader=training_loader,
                                                     unlabeled_loader=unlabeled_loader,
                                                     trained_model=trained_model,
                                                     batch_size=batch_size,
                                                     uncertainty_method=uncertainty_method)
    
    """
        Also selects instances from clusters based on UCB.
    """
    def _get_max_ucb_instance_from_cluster(self, cluster_cid, prev_selected_instances_from_cluster):
        instances_idx = np.where(self.clusters_unlabeled == cluster_cid)[0]
        instances_idx = np.setdiff1d(instances_idx, prev_selected_instances_from_cluster)
        X_instances = self.unlabeled_loader.get_features()[instances_idx]
        instances_preds = self.trained_model.predict(X_instances)[:,0]
        instances_uncertainty = self.trained_model.get_uncertainty(X=X_instances, 
                                                                   uncertainty_method=self.uncertainty_method,
                                                                   uncertainty_params_list=self.uncertainty_params_list)[:,0]
        instances_ucb = instances_uncertainty
        max_ucb_instance_argmax = np.argmax(instances_ucb)
        max_ucb_instance_idx = instances_idx[max_ucb_instance_argmax]
        
        # update cluster information
        # TODO: maybe bad design to put updating in this function; becuase it is unrelated to selecting max ucb instance.
        remaining_instances = np.setdiff1d(np.arange(len(instances_idx)), [max_ucb_instance_argmax])
        if len(remaining_instances) > 0:
            avg_cluster_uncertainty = np.nan_to_num(np.mean(instances_uncertainty[remaining_instances]))
            self.clusters_df.loc[cluster_cid, 'Mean Uncertainty'] = avg_cluster_uncertainty
            self.clusters_df.loc[cluster_cid, 'Cluster Mol Count'] = len(remaining_instances)
        else: # drop cluster if no more instances to select
            self.clusters_df = self.clusters_df.drop(cluster_cid, axis=0)
        return max_ucb_instance_idx
            
    """
        Uses UCB with mean of cluster/arm estimated by average model prediction. 
        Uncertainty measured by model uncertainty.
    """
    def select_next_batch(self):
        current_budget = self.batch_size
        selected_cluster_instances_dict = {}
        # populate self.clusters_df
        self._compute_cluster_uncertainty()
        
        # TODO: optimize recalculation of argmax since only means for the previously selected cluster is updated each iteration.
        while current_budget > 0:
            ucb_clusters = self.clusters_df['Mean Uncertainty']
            ucb_max_cid = self.clusters_df['Cluster ID'].iloc[np.argmax(ucb_clusters.values)]
            
            if ucb_max_cid not in selected_cluster_instances_dict:
                selected_cluster_instances_dict[ucb_max_cid] = []
            
            selected_instance_idx = self._get_max_ucb_instance_from_cluster(ucb_max_cid, selected_cluster_instances_dict[ucb_max_cid])
                    
            selected_cluster_instances_dict[ucb_max_cid].append(selected_instance_idx)
            current_budget -= 1
            
        selected_cluster_instances_pairs = [(k, selected_cluster_instances_dict[k]) for k in selected_cluster_instances_dict]
        return (selected_cluster_instances_pairs, []) # exploration selection is None in this case