from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ..utils.data_utils import get_duplicate_smiles, get_dissimilarity_matrix, feature_dist_func_dict

class NBSBase(object):
    """Abstract base Next Batch Selector class.
    # Properties
        training_loader: data loader for the training data.
        unlabeled_loader: data loader for unlabeled pool of instances.
        trained_model: model that is trained on the training data.
        next_batch_selector_params: config parameters for this batch selector.
    # Methods
        select_next_batch(): returns the next batch to be tested selected from the unlabeled pool.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 batch_size=384,
                 intra_cluster_dissimilarity_threshold=0.0,
                 feature_dist_func="tanimoto_dissimilarity"):
        self.training_loader = training_loader
        self.unlabeled_loader = unlabeled_loader
        self.trained_model = trained_model
        self.batch_size = batch_size
        self.intra_cluster_dissimilarity_threshold = intra_cluster_dissimilarity_threshold
        self.feature_dist_func = feature_dist_func_dict()[feature_dist_func]
        
        # remove already labeled molecules by checking training and unlabeled pool overlap
        # note duplicates determined via rdkit smiles
        smiles_train = training_loader.get_smiles()
        smiles_unlabeled = unlabeled_loader.get_smiles()
        idx_to_drop = get_duplicate_smiles(smiles_train, smiles_unlabeled)
        self.unlabeled_loader.idx_to_drop = idx_to_drop
        
        # throw exception if after dropping overlapping idx, there are no more unlabeled data to select
        if self.unlabeled_loader.get_size()[0] == 0:
            raise RuntimeError('Training data and unlabaled data overlap completely.\n' 
                               'This means there is no more unlabeled data to select from.')
    
    """
        Selects dissimilar instances given the instance indices. 
        Treats all instances identified by original_instance_idx as belonging to the same cluster.
        If useIntraClusterThreshold, only instances with avg dissimilarity >= self.intra_cluster_dissimilarity_threshold qualify. 
    """
    def _select_dissimilar_instances(self,
                                     original_instance_idx, 
                                     budget,
                                     useIntraClusterThreshold=True):
        selected_instances = []
        remaining_budget = budget
        features_unlabeled = self.unlabeled_loader.get_features()
        features_instances = features_unlabeled[original_instance_idx,:]
        
        intra_cluster_dissimilarity = get_dissimilarity_matrix(features_instances,
                                                               self.feature_dist_func)
        
        if remaining_budget > 0:
            # select instance with highest avg dissimilarity first
            avg_dissimilarity = np.mean(intra_cluster_dissimilarity, axis=0)
            curr_selected_idx = np.argsort(avg_dissimilarity)[::-1][0]
            selected_instances.append(curr_selected_idx)
            remaining_budget -= 1
        
        # select remaining instances based on what was already selected
        from functools import reduce
        while remaining_budget > 0:
            qualifying_idx = []
            if useIntraClusterThreshold:
                for idx in selected_instances:
                    qualifying_idx.append(np.where(intra_cluster_dissimilarity[idx,:] >= self.intra_cluster_dissimilarity_threshold)[0])
                
                qualifying_idx = reduce(np.intersect1d, qualifying_idx)
                if qualifying_idx.shape[0] == 0:
                    break
            else:
                qualifying_idx = range(intra_cluster_dissimilarity.shape[1])
                
            sub_matrix = intra_cluster_dissimilarity[selected_instances,:]
            avg_dissimilarity = np.mean(sub_matrix[:,qualifying_idx], axis=0)
            curr_selected_idx = qualifying_idx[np.argsort(avg_dissimilarity)[::-1][0]]
            selected_instances.append(curr_selected_idx)
            remaining_budget -= 1
        
        selected_instances = list(original_instance_idx[selected_instances])
        return selected_instances, remaining_budget

    """
        Selects instances from selected_cluster using a probability distribution without replacement. 
        If instance_proba=None, then selects instances uniformly.
    """
    def _select_random_instances(self,
                                 original_instance_idx, 
                                 budget,
                                 instance_proba=None):
        remaining_budget = budget
        selected_instances = list(np.random.choice(original_instance_idx, size=int(remaining_budget), 
                                                   replace=False, p=instance_proba))
        remaining_budget -= len(selected_instances)
        return selected_instances, remaining_budget
        
    """
        Simple budget allocation of taking the ratio of ee counts.
    """
    def _get_ee_budget(self, candidate_exploitation_instances_total, candidate_exploration_instances_total):
        exploitation_ratio = candidate_exploitation_instances_total / (candidate_exploitation_instances_total + candidate_exploration_instances_total) 
        exploitation_budget = np.floor(exploitation_ratio * self.batch_size)
        return exploitation_budget
        
    
    def select_next_batch(self):
        raise NotImplementedError