from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nbs_base import NBSBase
import pandas as pd
import numpy as np

class ClusterBasedSelector(NBSBase):
    """
    Selects next batch based on cluster information.
    See latest slides for more description.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 next_batch_selector_params):
        super(ClusterBasedSelector, self).__init__(training_loader,
                                                   unlabeled_loader,
                                                   trained_model,
                                                   next_batch_selector_params)
        self.batch_size = self.next_batch_selector_params["batch_size"]
        self.activity_threshold = self.next_batch_selector_params["activity_threshold"]
        self.exploitation_threshold = self.next_batch_selector_params["exploitation_threshold"]
        self.exploitation_alpha = self.next_batch_selector_params["exploitation_alpha"]
        self.exploration_threshold = self.next_batch_selector_params["exploration_threshold"]
        self.exploration_beta = self.next_batch_selector_params["exploration_beta"]
        self.dissimilarity_lambda = self.next_batch_selector_params["dissimilarity_lambda"]
        self.intra_cluster_dissimilarity_threshold = self.next_batch_selector_params["intra_cluster_dissimilarity_threshold"]
        
        # get clusters now since they are used in many calculations
        self.clusters_train = training_loader.get_clusters()
        self.clusters_unlabeled = unlabeled_loader.get_clusters()
        
        # create pandas df for various cluster calculations
        self.cluster_cols = ['Cluster ID', 'Cluster Mol Count',
                             'Density', 'Coverage', 
                             'Mean Uncertainty', 'Mean Activity Prediction',  'Mean Cost',
                             'High Activity Prediction Count',
                             'Exploitation Weight', 'Exploration Weight']
        self.clusters_df = pd.DataFrame(data=np.nan*np.zeros((self.total_clusters, len(self.cluster_cols)),
                                        columns=self.cluster_cols)
        self.clusters_df.iloc[:,[0,1]] = np.unique(np.hstack([self.clusters_train, self.clusters_unlabeled]),
                                                   return_counts=True)
        self.clusters_df.index = self.clusters_df['Cluster ID']
        self.total_clusters = len(self.clusters_df)
        
        # keep track of clusters selected already
        self.selected_exploitation_clusters = []
        self.selected_exploration_clusters = []
        
    def _compute_cluster_densities(self):
        total_molecule_count = np.sum(self.clusters_df['Cluster Mol Count'])
        for ci in self.clusters_df['Cluster ID']:
            density_i = self.clusters_df['Cluster Mol Count'].loc[ci] / total_molecule_count
            self.clusters_df.loc[ci, 'Density'] = density_i
    
    def _compute_cluster_coverage(self):
        for ci in self.clusters_df['Cluster ID']:
            cluster_labeled_count = np.sum(self.clusters_train == ci)
            coverage_i = cluster_labeled_count / self.clusters_df['Cluster Mol Count'].loc[ci]
            self.clusters_df.loc[ci, 'Coverage'] = coverage_i

    def _compute_cluster_uncertainty(self):
        preds_unlabeled = self.trained_model.predict(unlabeled_loader.get_features())
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            cluster_preds = preds_unlabeled[mol_idx]
            cluster_uncertainty = 1 - (np.abs(cluster_preds - 0.5) / 0.5)
            avg_cluster_uncertainty_i = np.mean(cluster_uncertainty)
            self.clusters_df.loc[ci, 'Mean Uncertainty'] = avg_cluster_uncertainty_i
            
    def _compute_cluster_activity_prediction(self):
        preds_unlabeled = self.trained_model.predict(unlabeled_loader.get_features())
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            cluster_preds = preds_unlabeled[mol_idx]
            cluster_preds = cluster_preds[cluster_preds >= self.activity_threshold]
            avg_cluster_activity_i = np.mean(cluster_preds)
            self.clusters_df.loc[ci, 'Mean Activity Prediction'] = avg_cluster_activity_i
            self.clusters_df.loc[ci, 'High Activity Prediction Count'] = len(cluster_preds)
            
    def _compute_cluster_cost(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            avg_cluster_cost_i = np.mean(costs_unlabeled[mol_idx])
            self.clusters_df.loc[ci, 'Mean Cost'] = avg_cluster_cost_i
    
    def _compute_cluster_exploitation_weight(self):
        self.clusters_df['Exploitation Weight'] = (self.exploitation_alpha * self.clusters_df['Mean Activity Prediction']) + \ 
                                                  ((1 - self.exploitation_alpha) * self.clusters_df['Coverage'] * self.clusters_df['Density'])
    
    def _compute_cluster_exploration_weight(self):
        self.clusters_df['Exploration Weight'] = (self.exploration_beta * self.clusters_df['Mean Uncertainty']) + \ 
                                                 ((1 - self.exploration_beta) * (1 - self.clusters_df['Coverage']))
                                                 
    def _get_candidate_exploitation_clusters(self):
        qualifying_exploitation_clusters = self.clusters_df['Exploitation Weight'] >= self.exploitation_threshold
        candidate_exploitation_clusters = self.cluster_df[qualifying_exploitation_clusters]['Cluster ID']
        return candidate_exploitation_clusters
        
    def _get_candidate_exploration_clusters(self):
        qualifying_exploration_clusters = self.clusters_df['Exploration Weight'] >= self.exploration_threshold
        candidate_exploration_clusters = self.cluster_df[qualifying_exploration_clusters]['Cluster ID']
        return candidate_exploration_clusters
    
    def _get_cluster_unlabeled_instances_count(self, cluster_ids):
        qualifying_clusters = self.clusters_df['Cluster ID'].isin(cluster_ids)
        cluster_unlabeled_instances_count = self.cluster_df[qualifying_clusters]['Cluster Mol Count'] * \
                                            (1 - self.cluster_df[qualifying_clusters]['Coverage'])
        return cluster_unlabeled_instances_count
    
    def _get_cluster_dissimilarity(self, selected_cluster_ids, candidate_cluster_ids):
        features_train = self.training_loader.get_features()
        features_unlabeled = self.unlabeled_loader.get_features()
        features_train_unlabeled = np.vstack([features_train, features_unlabeled])
        clusters_train_unlabeled = np.hstack([self.clusters_train, self.clusters_unlabeled])
        
        clusters_ordered_ids, clusters_avg_dissimilarity = get_avg_cluster_dissimilarity(clusters_train_unlabeled, 
                                                                                         features_train_unlabeled, 
                                                                                         selected_cluster_ids, 
                                                                                         candidate_cluster_ids)
        
        return clusters_ordered_ids, clusters_avg_dissimilarity
        
    """
        Currently not used. 
    """
    def _adjust_overlapping_clusters(self, candidate_exploitation_clusters, candidate_exploration_clusters):
        exploitation_clusters_to_drop = []
        exploration_clusters_to_drop = []
        overlapping_clusters = np.intersect1d(candidate_exploitation_clusters, candidate_exploration_clusters)
        for ci in overlapping_clusters:
            if self.cluster_df['Exploitation Weight'].loc[ci] >= self.cluster_df['Exploration Weight'].loc[ci]:
                exploration_clusters_to_drop.append(ci)
            else:
                exploitation_clusters_to_drop.append(ci)
                
        candidate_exploitation_clusters = candidate_exploitation_clusters[~candidate_exploitation_clusters.isin(exploitation_clusters_to_drop)]
        candidate_exploration_clusters = candidate_exploration_clusters[~candidate_exploration_clusters.isin(exploration_clusters_to_drop)]
        return candidate_exploitation_clusters, candidate_exploration_clusters
    
    """
        Simple budget allocation of taking the ratio of ee counts.
    """
    def _get_ee_budget(self, candidate_exploitation_instances_count, candidate_exploration_instances_count):
        exploitation_ratio = candidate_exploitation_instances_count / (candidate_exploitation_instances_count + candidate_exploration_instances_count) 
        exploitation_budget = np.floor(exploitation_ratio * self.batch_size)
        return exploitation_budget

    def _select_instances_from_clusters(self,
                                        candidate_clusters, 
                                        total_budget,
                                        weight_column='Exploitation Weight',
                                        useIntraClusterThreshold=True,
                                        useAdaptiveClusterBudget=False):
        selected_instances = []
        selected_clusters = []
        budget_per_cluster = np.zeros(shape=(len(candidate_clusters),))
        budget_per_cluster[:] = np.floor(total_budget / len(candidate_clusters))
        if useAdaptiveClusterBudget:
            cluster_unlabeled_counts = self._get_cluster_unlabeled_instances_count(candidate_clusters.values).values
            total_unlabeled_counts = np.sum(cluster_unlabeled_counts)
            budget_per_cluster[:] = np.floor(total_budget * (cluster_unlabeled_counts /  total_unlabeled_counts))
        budget_per_cluster[-1] = total_budget - np.sum(budget_per_cluster[:-1])
        
        remaining_total_budget = total_budget
        # select highest weighted cluster first
        cluster_weights = self.cluster_df[weight_column].loc[candidate_clusters.values].values
        curr_selected_cluster = candidate_clusters.values[np.argsort(cluster_weights)[::-1][0]]
        selected_instances_cluster, remaining_cluster_budget = self._select_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                  budget_per_cluster[0],
                                                                                                  useIntraClusterThreshold=useIntraClusterThreshold)
        selected_instances.extend(selected_instances_cluster)
        selected_clusters.append(curr_selected_cluster)
        remaining_total_budget -= len(selected_instances_cluster)
        
        # select remaining clusters based on what was already selected
        i=1
        while i < len(candidate_clusters) and remaining_total_budget > 0:
            budget_per_cluster[i] = budget_per_cluster[i-1] + remaining_cluster_budget
            rem_candidate_clusters = candidate_clusters.drop(selected_clusters)
            cluster_weights = self.cluster_df[weight_column].loc[rem_candidate_clusters.values].values
            _, clusters_avg_dissimilarity = self._get_cluster_dissimilarity(selected_clusters, 
                                                                            rem_candidate_clusters.values)
                                                                            
            adjusted_cluster_weights = self.dissimilarity_lambda * clusters_avg_dissimilarity + \
                                       ((1 - self.dissimilarity_lambda) * cluster_weights)
            curr_selected_cluster = rem_candidate_clusters.values[np.argsort(adjusted_cluster_weights)[::-1][0]]
            selected_instances_cluster, remaining_cluster_budget = self._select_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                              budget_per_cluster[i],
                                                                                                              useIntraClusterThreshold=useIntraClusterThreshold)
            selected_instances.extend(selected_instances_cluster)
            selected_clusters.append(curr_selected_cluster)
            remaining_total_budget -= len(selected_instances_cluster)
            i+=1
        
        return selected_instances, selected_clusters
    
    def _select_instances_from_single_cluster(self,
                                              selected_cluster, 
                                              cluster_budget,
                                              useIntraClusterThreshold=True):
        selected_instances = []
        remaining_cluster_budget = cluster_budget
        features_unlabeled = self.unlabeled_loader.get_features()
        original_instance_idx = np.where(self.clusters_unlabeled == selected_cluster)[0]
        features_cluster = features_unlabeled[original_instance_idx,:]
        
        intra_cluster_dissimilarity = get_dissimilarity_matrix(features_cluster)
        
        # select instance with highest avg dissimilarity first
        avg_dissimilarity = np.mean(intra_cluster_dissimilarity, axis=0)
        curr_selected_idx = np.argsort(avg_dissimilarity)[::-1][0]
        selected_instances.append(curr_selected_idx)
        remaining_cluster_budget -= 1
        
        # select remaining instances based on what was already selected
        from functools import reduce
        while remaining_cluster_budget > 0:
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
            remaining_cluster_budget -= 1
        
        selected_instances = list(original_instance_idx[selected_instances])
        return selected_instances, remaining_cluster_budget
        
    def select_next_batch(self):
        # populate self.cluster_df
        self._compute_cluster_densities()
        self._compute_cluster_coverage()
        self._compute_cluster_uncertainty()
        self._compute_cluster_activity_prediction()
        self._compute_cluster_cost()
        
        # compute cluster exploitation and exploration weights
        self._compute_cluster_exploitation_weight()
        self._compute_cluster_exploration_weight()
        
        # get qualifying candidate exploitation and exploration clusters
        candidate_exploitation_clusters = self._get_candidate_exploitation_clusters()
        candidate_exploration_clusters = self._get_candidate_exploration_clusters()        
        
        # get exploration and exploitation count estimates
        candidate_exploitation_instances_count = np.sum(self.clusters_df.loc[candidate_exploitation_clusters.values, 'High Activity Prediction Count'])
        candidate_exploration_instances_count = self._get_cluster_unlabeled_instances_count(candidate_exploration_clusters)
        
        # compute budget assigned to exploitation vs exploration
        exploitation_budget = self._get_ee_budget(candidate_exploitation_instances_count, 
                                                  candidate_exploration_instances_count)
        
        # start selecting exploitation instances from exploitation clusters
        selected_exploitation_instances, selected_exploitation_clusters = self._select_instances_from_clusters(candidate_exploitation_clusters, 
                                                                                                               exploitation_budget,
                                                                                                               weight_column='Exploitation Weight',
                                                                                                               useIntraClusterThreshold=True,
                                                                                                               useAdaptiveClusterBudget=False)
        
        # start selecting exploration instances from exploration clusters
        exploration_budget = self.batch_size - len(selected_exploitation_instances)
        update_exploration_clusters = np.setdiff1d(candidate_exploration_clusters, selected_exploitation_clusters)
        candidate_exploration_clusters = candidate_exploration_clusters[candidate_exploration_clusters.isin(update_exploration_clusters)]
        selected_exploration_instances, selected_exploration_clusters = self._select_instances_from_clusters(candidate_exploration_clusters, 
                                                                                                             exploration_budget,
                                                                                                             weight_column='Exploration Weight',
                                                                                                             useIntraClusterThreshold=False,
                                                                                                             useAdaptiveClusterBudget=True)
        
        