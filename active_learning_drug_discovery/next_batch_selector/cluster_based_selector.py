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
        self.exploration_threshold = self.next_batch_selector_params["exploration_threshold"]
        self.exploitation_threshold = self.next_batch_selector_params["exploitation_threshold"]
        
        # get clusters now since they are used in many calculations
        self.clusters_train = training_loader.get_clusters()
        self.clusters_unlabeled = unlabeled_loader.get_clusters()
        
        # create pandas df for various cluster calculations
        self.cluster_cols = ['Cluster ID', 'Cluster Mol Count',
                             'Density', 'Coverage', 
                             'Mean Uncertainty', 'Mean Activity Prediction', 
                             'Mean Dissimilarity', 'Mean Cost',
                             'Exploitation Weight', 'Exploration Weight']
        clusters_df = pd.DataFrame(data=np.zeros((self.total_clusters, len(self.cluster_cols)),
                                   columns=self.cluster_cols)
        clusters_df.iloc[:,[0,1]] = np.unique(np.hstack([self.clusters_train, self.clusters_unlabeled]),
                                              return_counts=True)
        self.total_clusters = len(self.clusters_df)
        
    def _compute_cluster_densities(self):
        total_molecule_count = np.sum(self.clusters_df['Cluster Mol Count'])
        for ci in range(self.total_clusters):
            density_i = self.clusters_df['Cluster Mol Count'].iloc[ci] / total_molecule_count
            clusters_df.loc[ci, 'Density'] = density_i
    
    def _compute_cluster_coverage(self):
        for ci in range(self.total_clusters):
            cluster_labeled_count = np.sum(self.clusters_train == self.clusters_df['Cluster ID'].iloc[ci])
            coverage_i = cluster_labeled_count / self.clusters_df['Cluster Mol Count'].iloc[ci]
            clusters_df.loc[ci, 'Coverage'] = coverage_i

    def _compute_cluster_uncertainty(self):
        preds_unlabeled = self.trained_model.predict(unlabeled_loader.get_features())
        for ci in range(self.total_clusters):
            mol_idx = np.where(self.clusters_unlabeled == self.clusters_df['Cluster ID'].iloc[ci])[0]
            cluster_preds = preds_unlabeled[mol_idx]
            cluster_uncertainty = 1 - (np.abs(cluster_preds - 0.5) / 0.5)
            avg_cluster_uncertainty_i = np.mean(cluster_uncertainty)
            clusters_df.loc[ci, 'Mean Uncertainty'] = avg_cluster_uncertainty_i
            
    def _compute_cluster_activity_prediction(self):
        preds_unlabeled = self.trained_model.predict(unlabeled_loader.get_features())
        for ci in range(self.total_clusters):
            mol_idx = np.where(self.clusters_unlabeled == self.clusters_df['Cluster ID'].iloc[ci])[0]
            cluster_preds = preds_unlabeled[mol_idx]
            cluster_preds = cluster_preds[cluster_preds >= self.activity_threshold]
            avg_cluster_activity_i = np.mean(cluster_preds)
            clusters_df.loc[ci, 'Mean Activity Prediction'] = avg_cluster_activity_i

    def _compute_cluster_dissimilarity(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in range(self.total_clusters):
            mol_idx = np.where(self.clusters_unlabeled == self.clusters_df['Cluster ID'].iloc[ci])[0]
            cluster_dissimilarity_i = np.mean(costs_unlabeled[mol_idx])
            clusters_df.loc[ci, 'Mean Dissimilarity'] = cluster_dissimilarity_i
            
    def _compute_cluster_cost(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in range(self.total_clusters):
            mol_idx = np.where(self.clusters_unlabeled == self.clusters_df['Cluster ID'].iloc[ci])[0]
            avg_cluster_cost_i = np.mean(costs_unlabeled[mol_idx])
            clusters_df.loc[ci, 'Mean Cost'] = avg_cluster_cost_i
    
    def _compute_cluster_exploitation_weight(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in range(self.total_clusters):
            mol_idx = np.where(self.clusters_unlabeled == self.clusters_df['Cluster ID'].iloc[ci])[0]
            avg_cluster_cost_i = np.mean(costs_unlabeled[mol_idx])
            clusters_df.loc[ci, 'Exploitation Weight'] = avg_cluster_cost_i

    def _compute_cluster_exploration_weight(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in range(self.total_clusters):
            mol_idx = np.where(self.clusters_unlabeled == self.clusters_df['Cluster ID'].iloc[ci])[0]
            avg_cluster_cost_i = np.mean(costs_unlabeled[mol_idx])
            clusters_df.loc[ci, 'Exploration Weight'] = avg_cluster_cost_i
            
    def select_next_batch(self):
        # populate self.cluster_df
        self._compute_cluster_densities()
        self._compute_cluster_coverage()
        self._compute_cluster_uncertainty()
        self._compute_cluster_activity_prediction()
        self._compute_cluster_dissimilarity()
        self._compute_cluster_cost()
        
        # compute cluster exploitation and exploration weights
        self._compute_cluster_exploitation_weight()
        self._compute_cluster_exploration_weight()
        