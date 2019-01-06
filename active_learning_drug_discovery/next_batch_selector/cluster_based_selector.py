"""
    Contains classes for cluster based selectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nbs_base import NBSBase
import pandas as pd
import numpy as np

class ClusterBasedSelector(NBSBase):
    """
    Base class for next batch selectors based on cluster information.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 batch_size=384,
                 intra_cluster_dissimilarity_threshold=0.0):
        super(ClusterBasedSelector, self).__init__(training_loader,
                                                   unlabeled_loader,
                                                   trained_model,
                                                   batch_size,
                                                   intra_cluster_dissimilarity_threshold)
        # get clusters now since they are used in many calculations
        self.clusters_train = training_loader.get_clusters()
        self.clusters_unlabeled = unlabeled_loader.get_clusters()
        
        # keep track of clusters selected already
        self.selected_exploitation_clusters = []
        self.selected_exploration_clusters = []
    
    def _get_avg_cluster_dissimilarity(self, selected_cluster_ids, candidate_cluster_ids):
        features_train = self.training_loader.get_features()
        features_unlabeled = self.unlabeled_loader.get_features()
        features_train_unlabeled = np.vstack([features_train, features_unlabeled])
        clusters_train_unlabeled = np.hstack([self.clusters_train, self.clusters_unlabeled])
        
        clusters_ordered_ids, avg_cluster_dissimilarity = get_avg_cluster_dissimilarity(clusters_train_unlabeled, 
                                                                                        features_train_unlabeled, 
                                                                                        selected_cluster_ids, 
                                                                                        candidate_cluster_ids)
        
        return clusters_ordered_ids, avg_cluster_dissimilarity
    
    def _get_candidate_exploitation_clusters(self):
        return None
        
    def _get_candidate_exploration_clusters(self):
        return None
    
    def _get_candidate_exploitation_instances_total(self, cluster_ids):
        return None
    
    def _get_candidate_exploration_instances_total(self, cluster_ids):
        exploration_instances_total = np.sum(self._get_candidate_exploration_instances_per_cluster_count(cluster_ids))
        return exploration_instances_total
        
     def _get_candidate_exploration_instances_per_cluster_count(self, cluster_ids):
        clusters_idx = np.in1d(self.clusters_unlabeled, cluster_ids)
        _, candidate_exploration_instances_per_cluster = np.unique(self.clusters_unlabeled[clusters_idx], 
                                                                   return_counts=True)
        candidate_exploration_instances_per_cluster = candidate_exploration_instances_per_cluster[np.argsort(cluster_ids)]
        return candidate_exploration_instances_per_cluster
        
    def _select_instances_from_clusters(self,
                                        candidate_clusters, 
                                        total_budget,
                                        useExploitationStrategy=True):
        return None 
        
    """
        Selects dissimilar instances from selected_cluster. 
    """
    def _select_dissimilar_instances_from_single_cluster(self,
                                                         selected_cluster, 
                                                         cluster_budget,
                                                         useIntraClusterThreshold=True):
        cluster_instance_idx = np.where(self.clusters_unlabeled == selected_cluster)[0]
        selected_cluster_instances, remaining_cluster_budget = self._select_dissimilar_instances(cluster_instance_idx, 
                                                                                                 cluster_budget,
                                                                                                 useIntraClusterThreshold=useIntraClusterThreshold)
        return selected_cluster_instances, remaining_cluster_budget
        
    """
        Selects instances from selected_cluster using a probability distribution without replacement. 
        If instance_proba=None, then selects instances uniformly.
    """
    def _select_random_instances_from_single_cluster(self,
                                                     selected_cluster, 
                                                     cluster_budget,
                                                     instance_proba=None):
        cluster_instance_idx = np.where(self.clusters_unlabeled == selected_cluster)[0]
        selected_cluster_instances, remaining_cluster_budget = self._select_random_instances(cluster_instance_idx, 
                                                                                             cluster_budget,
                                                                                             instance_proba=instance_proba)
        return selected_cluster_instances, remaining_cluster_budget
        
    def select_next_batch(self):
        # get qualifying candidate exploitation and exploration clusters
        candidate_exploitation_clusters = self._get_candidate_exploitation_clusters()
        candidate_exploration_clusters = self._get_candidate_exploration_clusters()        
        
        # get exploration and exploitation count estimates
        candidate_exploitation_instances_total = self._get_candidate_exploitation_instances_total(candidate_exploitation_clusters)
        candidate_exploration_instances_total = self._get_candidate_exploration_instances_total(candidate_exploration_clusters)
        
        # compute budget assigned to exploitation vs exploration
        exploitation_budget = self._get_ee_budget(candidate_exploitation_instances_total, 
                                                  candidate_exploration_instances_total)
        
        # start selecting exploitation instances from exploitation clusters
        selected_exploitation_cluster_instances_pairs = self._select_instances_from_clusters(candidate_exploitation_clusters, 
                                                                                             exploitation_budget, 
                                                                                             useExploitationStrategy=True)
        
        # start selecting exploration instances from exploration clusters
        exploration_budget = self.batch_size - len(selected_exploitation_instances)
        update_exploration_clusters = np.setdiff1d(candidate_exploration_clusters, selected_exploitation_clusters)
        candidate_exploration_clusters = update_exploration_clusters
        selected_exploration_cluster_instances_pairs = self._select_instances_from_clusters(candidate_exploration_clusters, 
                                                                                            exploration_budget, 
                                                                                            useExploitationStrategy=False)
        
        return (selected_exploitation_cluster_instances_pairs, 
                selected_exploration_cluster_instances_pairs)
                
class ClusterBasedWCSelector(ClusterBasedSelector):
    """
    Selects next batch based on cluster information.
    Weighted-Clusters (WC): Uses a exploitation-exploration weighting scheme to select among clusters, 
                            and then selects instances within a cluster either based on dissimlarity or randomness. (see params)
    See latest slides for more description.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 batch_size=384,
                 intra_cluster_dissimilarity_threshold=0.0,
                 select_dissimilar_instances_within_cluster=True,
                 exploitation_activity_threshold=0.75,
                 exploitation_threshold=0.5,
                 exploitation_alpha=0.5,
                 exploitation_dissimilarity_lambda=0.5,
                 use_intra_cluster_threshold_for_exploitation=True,
                 use_proportional_cluster_budget_for_exploitation=False,
                 exploration_strategy="weighted",
                 exploration_threshold=0.5,
                 exploration_beta=0.5,
                 exploration_dissimilarity_lambda=0.5,
                 use_intra_cluster_threshold_for_exploration=False,
                 use_proportional_cluster_budget_for_exploration=True):
        super(ClusterBasedWCSelector, self).__init__(training_loader,
                                                     unlabeled_loader,
                                                     trained_model,
                                                     batch_size,
                                                     intra_cluster_dissimilarity_threshold)
        self.select_dissimilar_instances_within_cluster = select_dissimilar_instances_within_cluster
        
        self.exploitation_activity_threshold = exploitation_activity_threshold
        self.exploitation_threshold = exploitation_threshold
        self.exploitation_alpha = exploitation_alpha
        self.exploitation_dissimilarity_lambda = exploitation_dissimilarity_lambda
        self.use_intra_cluster_threshold_for_exploitation = use_intra_cluster_threshold_for_exploitation
        self.use_proportional_cluster_budget_for_exploitation = use_proportional_cluster_budget_for_exploitation
        
        self.exploration_strategy = exploration_strategy
        self.exploration_threshold = exploration_threshold
        self.exploration_beta = exploration_beta
        self.exploration_dissimilarity_lambda = exploration_dissimilarity_lambda
        self.use_intra_cluster_threshold_for_exploration = use_intra_cluster_threshold_for_exploration
        self.use_proportional_cluster_budget_for_exploration = use_proportional_cluster_budget_for_exploration
        
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
            cluster_preds = cluster_preds[cluster_preds >= self.exploitation_activity_threshold]
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
        if self.exploration_strategy == "weighted":
            self.clusters_df['Exploration Weight'] = (self.exploration_beta * self.clusters_df['Mean Uncertainty']) + \ 
                                                     ((1 - self.exploration_beta) * (1 - self.clusters_df['Coverage']))
        elif self.exploration_strategy == "dissimilar":
            cluster_ids = self.cluster_df['Cluster ID'].values
            _, avg_cluster_dissimilarity = self._get_avg_cluster_dissimilarity(cluster_ids, 
                                                                               cluster_ids)
            self.clusters_df['Exploration Weight'] = avg_cluster_dissimilarity
            
    def _get_candidate_exploitation_clusters(self):
        qualifying_exploitation_clusters = self.clusters_df['Exploitation Weight'] >= self.exploitation_threshold
        candidate_exploitation_clusters = self.cluster_df[qualifying_exploitation_clusters]['Cluster ID'].values
        return candidate_exploitation_clusters
        
    def _get_candidate_exploration_clusters(self):
        if self.exploration_strategy == "weighted":
            qualifying_exploration_clusters = self.clusters_df['Exploration Weight'] >= self.exploration_threshold
            candidate_exploration_clusters = self.cluster_df[qualifying_exploration_clusters]['Cluster ID'].values
        elif self.exploration_strategy == "random":
            candidate_exploration_clusters = np.unique(self.clusters_unlabeled)
        else: # self.exploration_strategy == "dissimilar"
            candidate_exploration_clusters = self.cluster_df['Cluster ID'].values
        return candidate_exploration_clusters
    
    def _get_candidate_exploitation_instances_total(self, cluster_ids):
        exploitation_instances_total = np.sum(self.clusters_df.loc[cluster_ids.values, 'High Activity Prediction Count'])
        return exploitation_instances_total
        
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
    
    def _select_instances_from_clusters(self,
                                        candidate_clusters, 
                                        total_budget,
                                        useExploitationStrategy=True):
        selected_instances_cluster, remaining_cluster_budget = None, None
        if useExploitationStrategy:
            selected_clusters_instances_pairs = self._select_instances_from_clusters_helper(candidate_exploitation_clusters, 
                                                                                            exploitation_budget, 
                                                                                            self.exploitation_dissimilarity_lambda,
                                                                                            weight_column='Exploitation Weight',
                                                                                            useIntraClusterThreshold=self.use_intra_cluster_threshold_for_exploitation,
                                                                                            useProportionalClusterBudget=self.use_proportional_cluster_budget_for_exploitation,
                                                                                            selectDissimilarInstancesWithinCluster=self.select_dissimilar_instances_within_cluster)
        else:
            selected_clusters_instances_pairs = self._select_instances_from_clusters(candidate_exploration_clusters, 
                                                                                     exploration_budget,
                                                                                     self.exploration_dissimilarity_lambda,
                                                                                     weight_column='Exploration Weight',
                                                                                     useIntraClusterThreshold=self.use_intra_cluster_threshold_for_exploration,
                                                                                     useProportionalClusterBudget=self.use_proportional_cluster_budget_for_exploration,
                                                                                     selectDissimilarInstancesWithinCluster=self.select_dissimilar_instances_within_cluster)
        return selected_clusters_instances_pairs
        
    """
        Helper method for selecting clusters using weight scheme.
    """
    def _select_instances_from_clusters_weighted(self,
                                                 candidate_clusters, 
                                                 total_budget,
                                                 dissimilarity_lambda,
                                                 weight_column='Exploitation Weight',
                                                 useIntraClusterThreshold=True,
                                                 useProportionalClusterBudget=False,
                                                 selectDissimilarInstancesWithinCluster=True):
        selected_clusters_instances_pairs = []
        curr_cluster_budget =  np.ceil(total_budget / len(candidate_clusters))
        if useProportionalClusterBudget:
            cluster_unlabeled_counts = self._get_candidate_exploration_instances_per_cluster_count(candidate_clusters)
            total_unlabeled_counts = np.sum(cluster_unlabeled_counts)
        
        remaining_total_budget = total_budget
        if remaining_total_budget > 0:
            # select highest weighted cluster first
            cluster_weights = self.cluster_df[weight_column].loc[candidate_clusters].values
            curr_selected_cluster_idx = np.argsort(cluster_weights)[::-1][0]
            curr_selected_cluster = candidate_clusters[curr_selected_cluster_idx]
            
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts))
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                                             curr_cluster_budget,
                                                                                                                             useIntraClusterThreshold=useIntraClusterThreshold)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                                         curr_cluster_budget)
            selected_clusters_instances_pairs.append((curr_selected_cluster,))
            selected_clusters_instances_pairs[-1] = selected_clusters_instances_pairs[-1] + (selected_instances_cluster,)
            remaining_total_budget -= len(selected_instances_cluster)
        
        # select remaining clusters based on what was already selected
        i=1
        while i < len(candidate_clusters) and remaining_total_budget > 0:
            selected_clusters_so_far = [x[0] for x in selected_clusters_instances_pairs]
            
            rem_candidate_clusters = np.setdiff1d(candidate_clusters, selected_clusters_so_far)
            cluster_weights = self.cluster_df[weight_column].loc[rem_candidate_clusters].values
            _, avg_cluster_dissimilarity = self._get_avg_cluster_dissimilarity(selected_clusters_so_far, 
                                                                               rem_candidate_clusters)
            # adjust cluster weights to include avg cluster dissimilarity
            adjusted_cluster_weights = dissimilarity_lambda * avg_cluster_dissimilarity + \
                                       ((1 - dissimilarity_lambda) * cluster_weights)
            curr_selected_cluster_idx = np.where(candidate_clusters == rem_candidate_clusters[np.argsort(adjusted_cluster_weights)[::-1][0]])[0]
            curr_selected_cluster = candidate_clusters[curr_selected_cluster_idx]
            
            # process current cluster budget
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts)) 
            curr_cluster_budget = curr_cluster_budget + remaining_cluster_budget
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                                             curr_cluster_budget,
                                                                                                                             useIntraClusterThreshold=useIntraClusterThreshold)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                                         curr_cluster_budget)
            selected_clusters_instances_pairs.append((curr_selected_cluster,))
            selected_clusters_instances_pairs[-1] = selected_clusters_instances_pairs[-1] + (selected_instances_cluster,)
            remaining_total_budget -= len(selected_instances_cluster)
            i+=1
        
        return selected_clusters_instances_pairs
    
    """
        Helper method for selecting clusters in a random manner.
    """
    def _select_instances_from_clusters_random(self,
                                               candidate_clusters, 
                                               total_budget,
                                               useProportionalClusterBudget=False,
                                               selectDissimilarInstancesWithinCluster=True):
        selected_clusters_instances_pairs = []
        curr_cluster_budget =  np.ceil(total_budget / len(candidate_clusters))
        if useProportionalClusterBudget:
            cluster_unlabeled_counts = self._get_candidate_exploration_instances_per_cluster_count(candidate_clusters)
            total_unlabeled_counts = np.sum(cluster_unlabeled_counts)
        
        remaining_total_budget = total_budget
        i=0
        rem_clusters_idx = list(np.arange(len(candidate_clusters)))
        while i < len(candidate_clusters) and remaining_total_budget > 0:
            curr_selected_cluster_idx = np.random.choice(rem_clusters_idx, size=1, replace=False)
            rem_clusters_idx = rem_clusters_idx.remove(curr_selected_cluster_idx)
            curr_selected_cluster = candidate_clusters[curr_selected_cluster_idx]
            
            # process current cluster budget
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts)) 
            curr_cluster_budget = curr_cluster_budget + remaining_cluster_budget
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                                             curr_cluster_budget,
                                                                                                                             useIntraClusterThreshold=False)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances_from_single_cluster(curr_selected_cluster, 
                                                                                                                         curr_cluster_budget)
            selected_clusters_instances_pairs.append((curr_selected_cluster,))
            selected_clusters_instances_pairs[-1] = selected_clusters_instances_pairs[-1] + (selected_instances_cluster,)
            remaining_total_budget -= len(selected_instances_cluster)
            i+=1
            
        return selected_clusters_instances_pairs
        
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
        
        return super(ClusterBasedWCSelector, self).select_next_batch()