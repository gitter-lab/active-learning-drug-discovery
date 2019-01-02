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
        
        # get clusters now since they are used in many calculations
        self.clusters_train = training_loader.get_clusters()
        self.clusters_unlabeled = unlabeled_loader.get_clusters()
        
        # create pandas df for various cluster calculations
        self.cluster_cols = ['Cluster ID', 'Cluster Mol Count',
                             'Density', 'Coverage', 
                             'Mean Uncertainty', 'Mean Activity Prediction', 
                             'Mean Dissimilarity', 'Mean Cost',
                             'Exploitation Weight', 'Exploration Weight']
        self.clusters_df = pd.DataFrame(data=np.zeros((self.total_clusters, len(self.cluster_cols)),
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

    def _compute_cluster_dissimilarity(self, selected_cluster_ids, candidate_cluster_ids):
        features_train = self.training_loader.get_features()
        features_unlabeled = self.unlabeled_loader.get_features()
        features_train_unlabeled = np.vstack([features_train, features_unlabeled])
        clusters_train_unlabeled = np.hstack([self.clusters_train, self.clusters_unlabeled])
        
        clusters_ordered_ids, clusters_avg_dissimilarity = get_avg_cluster_dissimilarity(clusters_train_unlabeled, 
                                                                                         features_train_unlabeled, 
                                                                                         selected_cluster_ids, 
                                                                                         candidate_cluster_ids)
        
        
        self.clusters_df.loc[clusters_ordered_ids, 'Mean Dissimilarity'] = clusters_avg_dissimilarity
            
    def _compute_cluster_cost(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            avg_cluster_cost_i = np.mean(costs_unlabeled[mol_idx])
            self.clusters_df.loc[ci, 'Mean Cost'] = avg_cluster_cost_i
    
    def _compute_cluster_exploitation_weight(self):
        self.clusters_df['Exploitation Weight'] = (self.exploitation_alpha * self.clusters_df['Mean Activity Prediction']) + \ 
                                                  ((1 - self.exploitation_alpha) * self.clusters_df['Coverage'] * self.clusters_df['Density'])
                                                 
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
        
    def _compute_cluster_exploration_weight(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == self.clusters_df['Cluster ID'].iloc[ci])[0]
            avg_cluster_cost_i = np.mean(costs_unlabeled[mol_idx])
            self.clusters_df.loc[ci, 'Exploration Weight'] = avg_cluster_cost_i
    
    def _adjust_overlapping_clusters(self, candidate_exploitation_clusters, candidate_exploration_clusters)
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
    def _compute_ee_budget(self, candidate_exploitation_instances_count, candidate_exploration_instances_count):
        exploitation_ratio = candidate_exploitation_instances_count / (candidate_exploitation_instances_count + candidate_exploration_instances_count) 
        exploitation_budget = np.floor(exploitation_ratio * self.batch_size)
        return exploitation_budget
    

    def _select_instances_from_clusters(candidate_clusters, 
                                        budget,
                                        useEntireBudget=True):
        return selected_instances
        
    def select_next_batch(self):
        # populate self.cluster_df
        self._compute_cluster_densities()
        self._compute_cluster_coverage()
        self._compute_cluster_uncertainty()
        self._compute_cluster_activity_prediction()
        self._compute_cluster_cost()
        
        # specify which cluster dissimilarities to compute
        selected_cluster_ids = self.cluster_df['Cluster ID'].values
        candidate_cluster_ids = self.cluster_df['Cluster ID'].values
        self._compute_cluster_dissimilarity(selected_cluster_ids, candidate_cluster_ids)
        
        # compute cluster exploitation and exploration weights
        self._compute_cluster_exploitation_weight()
        self._compute_cluster_exploration_weight()
        
        # get qualifying candidate exploitation and exploration clusters
        candidate_exploitation_clusters = self._get_candidate_exploitation_clusters()
        candidate_exploration_clusters = self._get_candidate_exploration_clusters()
        
        # check for overlapping clusters between ee and assign based on weight test
        candidate_exploitation_clusters, candidate_exploration_clusters = self._adjust_overlapping_clusters(candidate_exploitation_clusters, 
                                                                                                            candidate_exploration_clusters)
        
        
        # get exploration and exploitation counts
        candidate_exploitation_instances_count = self._get_cluster_unlabeled_instances_count(candidate_exploitation_clusters)
        candidate_exploration_instances_count = self._get_cluster_unlabeled_instances_count(candidate_exploration_clusters)
        
        # compute budget assigned to exploitation vs exploration
        exploitation_budget = self._compute_ee_budget(candidate_exploitation_instances_count, candidate_exploration_instances_count)
        
        # start selecting exploitation instances from exploitation clusters
        selected_exploitation_instances = self._select_instances_from_clusters(candidate_exploitation_clusters, 
                                                                               exploitation_budget,
                                                                               useEntireBudget=False)
        
        # start selecting exploration instances from exploration clusters
        exploration_budget = self.batch_size - len(selected_exploitation_instances)
        selected_exploration_instances = self._select_instances_from_clusters(candidate_exploration_clusters, 
                                                                              exploration_budget,
                                                                              useEntireBudget=True)
        
        