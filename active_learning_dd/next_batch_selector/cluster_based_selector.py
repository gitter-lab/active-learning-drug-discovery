"""
    Contains classes for cluster based selectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nbs_base import NBSBase
from ..utils.data_utils import get_avg_cluster_dissimilarity, get_avg_cluster_dissimilarity_from_file
import pandas as pd
import numpy as np
import time

class ClusterBasedSelector(NBSBase):
    """
    Base class for next batch selectors based on cluster information.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 batch_size=384,
                 intra_cluster_dissimilarity_threshold=0.0,
                 feature_dist_func="tanimoto_dissimilarity",
                 dissimilarity_memmap_filename=None,
                 use_consensus_distance=False):
        super(ClusterBasedSelector, self).__init__(training_loader,
                                                   unlabeled_loader,
                                                   trained_model,
                                                   batch_size,
                                                   intra_cluster_dissimilarity_threshold,
                                                   feature_dist_func=feature_dist_func,
                                                   dissimilarity_memmap_filename=dissimilarity_memmap_filename)
        # get clusters now since they are used in many calculations
        self.clusters_train = self.training_loader.get_clusters()
        self.clusters_unlabeled = self.unlabeled_loader.get_clusters()
        
        # keep track of clusters selected already
        self.selected_exploitation_clusters = []
        self.selected_exploration_clusters = []
        
        self.use_consensus_distance = use_consensus_distance
    
    def _get_avg_cluster_dissimilarity(self, selected_cluster_ids, candidate_cluster_ids):
        clusters_train_unlabeled = np.hstack([self.clusters_train, self.clusters_unlabeled])
        
        if self.dissimilarity_memmap_filename is None:
            train_size = self.training_loader.get_size()[0]
            cid_instances = np.in1d(clusters_train_unlabeled, np.hstack([selected_cluster_ids, candidate_cluster_ids]))
            clusters_train_unlabeled = clusters_train_unlabeled[cid_instances]
            
            # read features
            cid_instances_train = cid_instances[:train_size]
            cid_instances_unlabeled = cid_instances[train_size:]
            
            # slice out targeted cluster instances only
            features_train = self.training_loader.get_features(cid_instances_train)
            features_unlabeled = self.unlabeled_loader.get_features(cid_instances_unlabeled)
            if features_train.shape[0] > 0:
                features_train_unlabeled = np.vstack([features_train, features_unlabeled])
            else: # case for empty training set
                features_train_unlabeled = features_unlabeled
            del features_train, features_unlabeled, cid_instances
            
            if self.use_consensus_distance:
                consensus_features = np.zeros(shape=(len(selected_cluster_ids)+len(candidate_cluster_ids), 
                                                     features_train_unlabeled.shape[1]))
                consensus_clusters = np.zeros(shape=(len(selected_cluster_ids)+len(candidate_cluster_ids),))
                for i, cid in enumerate(np.hstack([selected_cluster_ids, candidate_cluster_ids])):
                    cid_instances = np.where(clusters_train_unlabeled == cid)[0]
                    cluster_features = features_train_unlabeled[cid_instances,:]
                    consensus_features[i,:] = ((np.sum(cluster_features, axis=0) / cluster_features.shape[0]) >= 0.5).astype(float)
                    consensus_clusters[i] = cid
                    
                features_train_unlabeled = consensus_features
                clusters_train_unlabeled = consensus_clusters
            
            clusters_ordered_ids, avg_cluster_dissimilarity = get_avg_cluster_dissimilarity(clusters_train_unlabeled, 
                                                                                            features_train_unlabeled, 
                                                                                            selected_cluster_ids, 
                                                                                            candidate_cluster_ids,
                                                                                            feature_dist_func=self.feature_dist_func)
        else:
            idx_ids_train_unlabeled = np.hstack([self.training_loader.get_idx_ids(), 
                                                 self.unlabeled_loader.get_idx_ids()])
            sorted_idx_ids = np.argsort(idx_ids_train_unlabeled)
            clusters_train_unlabeled = clusters_train_unlabeled[sorted_idx_ids]
            clusters_ordered_ids, avg_cluster_dissimilarity = get_avg_cluster_dissimilarity_from_file(clusters_train_unlabeled, 
                                                                                                      self.dissimilarity_memmap_filename, 
                                                                                                      len(clusters_train_unlabeled),
                                                                                                      selected_cluster_ids, 
                                                                                                      candidate_cluster_ids)
                                                                                        
        return clusters_ordered_ids, avg_cluster_dissimilarity
    
    def _get_candidate_exploitation_clusters(self):
        raise NotImplementedError
        
    def _get_candidate_exploration_clusters(self):
        raise NotImplementedError
    
    def _get_candidate_exploitation_instances_total(self, cluster_ids):
        raise NotImplementedError
    
    def _get_candidate_exploration_instances_total(self, cluster_ids):
        exploration_instances_total = np.sum(self._get_candidate_exploration_instances_per_cluster_count(cluster_ids))
        return exploration_instances_total
        
    def _get_candidate_exploration_instances_per_cluster_count(self, cluster_ids):
        clusters_idx = np.in1d(self.clusters_unlabeled, cluster_ids)
        u, candidate_exploration_instances_per_cluster = np.unique(self.clusters_unlabeled[clusters_idx], 
                                                                   return_counts=True)
        sorted_idx = np.argsort(cluster_ids)
        rev_sorted_idx = np.zeros(len(u), dtype=int)
        rev_sorted_idx[sorted_idx] = np.arange(len(u)) # adapted from: https://stackoverflow.com/a/10831155
        candidate_exploration_instances_per_cluster = candidate_exploration_instances_per_cluster[rev_sorted_idx]
        return candidate_exploration_instances_per_cluster
        
    def _select_instances_from_clusters(self,
                                        candidate_clusters, 
                                        total_budget,
                                        useExploitationStrategy=True):
        raise NotImplementedError
        
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
        start_time=time.time()
        candidate_exploitation_clusters = self._get_candidate_exploitation_clusters()
        print("Done _get_candidate_exploitation_clusters. Took {} seconds. size: {}".format(time.time()-start_time, candidate_exploitation_clusters.shape))
        start_time=time.time()
        candidate_exploration_clusters = self._get_candidate_exploration_clusters()
        print("Done _get_candidate_exploration_clusters. Took {} seconds. size: {}".format(time.time()-start_time, candidate_exploration_clusters.shape))

        # get exploration and exploitation count estimates
        start_time=time.time()
        candidate_exploitation_instances_total = self._get_candidate_exploitation_instances_total(candidate_exploitation_clusters)
        print("Done _get_candidate_exploitation_instances_total: {}. Took {} seconds.".format(candidate_exploitation_instances_total, time.time()-start_time))
        start_time=time.time()
        candidate_exploration_instances_total = self._get_candidate_exploration_instances_total(candidate_exploration_clusters)
        print("Done _get_candidate_exploration_instances_total: {}. Took {} seconds.".format(candidate_exploration_instances_total, time.time()-start_time))    

        # compute budget assigned to exploitation vs exploration
        exploitation_budget = self._get_ee_budget(candidate_exploitation_instances_total, 
                                                  candidate_exploration_instances_total)

        # start selecting exploitation instances from exploitation clusters
        start_time=time.time()
        selected_exploitation_cluster_instances_pairs = self._select_instances_from_clusters(candidate_exploitation_clusters, 
                                                                                             exploitation_budget, 
                                                                                             useExploitationStrategy=True)
        print("Done _select_instances_from_clusters-candidate_exploitation_clusters: {}. Took {} seconds.".format(sum([len(x[1]) for x in selected_exploitation_cluster_instances_pairs]), 
                                                                                                                  time.time()-start_time))
        # start selecting exploration instances from exploration clusters
        # update exploration_budget 
        start_time=time.time()
        selected_exploitation_clusters = []
        selected_exploitation_instances_count = 0
        if len(selected_exploitation_cluster_instances_pairs) > 0:
            selected_exploitation_clusters = [x[0] for x in selected_exploitation_cluster_instances_pairs]
            selected_exploitation_instances = [x[1] for x in selected_exploitation_cluster_instances_pairs]
            selected_exploitation_instances_count = np.hstack(selected_exploitation_instances).shape[0]
        exploration_budget = self.batch_size - selected_exploitation_instances_count
        update_exploration_clusters = np.setdiff1d(candidate_exploration_clusters, selected_exploitation_clusters)
        candidate_exploration_clusters = update_exploration_clusters
            
        selected_exploration_cluster_instances_pairs = self._select_instances_from_clusters(candidate_exploration_clusters, 
                                                                                            exploration_budget, 
                                                                                            useExploitationStrategy=False)
        print("Done _select_instances_from_clusters-candidate_exploration_clusters: {}. Took {} seconds.".format(sum([len(x[1]) for x in selected_exploration_cluster_instances_pairs]),
                                                                                                                 time.time()-start_time))
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
                 feature_dist_func="tanimoto_dissimilarity",
                 dissimilarity_memmap_filename=None,
                 use_consensus_distance=False,
                 uncertainty_method="least_confidence",
                 select_dissimilar_instances_within_cluster=True,
                 exploitation_use_quantile_for_activity=False,
                 exploitation_sample_actives_from_clusters=False,
                 exploitation_activity_threshold=0.75,
                 exploitation_use_quantile_for_weight=False,
                 exploitation_weight_threshold=0.5,
                 exploitation_alpha=0.5,
                 exploitation_dissimilarity_lambda=0.5,
                 use_intra_cluster_threshold_for_exploitation=True,
                 use_proportional_cluster_budget_for_exploitation=False,
                 exploration_strategy="weighted",
                 exploration_use_quantile_for_weight=False,
                 exploration_weight_threshold=0.5,
                 exploration_beta=0.5,
                 exploration_dissimilarity_lambda=0.5,
                 use_intra_cluster_threshold_for_exploration=False,
                 use_proportional_cluster_budget_for_exploration=True):
        super(ClusterBasedWCSelector, self).__init__(training_loader,
                                                     unlabeled_loader,
                                                     trained_model,
                                                     batch_size,
                                                     intra_cluster_dissimilarity_threshold,
                                                     feature_dist_func=feature_dist_func,
                                                     dissimilarity_memmap_filename=dissimilarity_memmap_filename,
                                                     use_consensus_distance=use_consensus_distance)
        self.select_dissimilar_instances_within_cluster = select_dissimilar_instances_within_cluster
        self.uncertainty_method = uncertainty_method
        self.uncertainty_params_list = None
        if isinstance(uncertainty_method, list):
            self.uncertainty_method = uncertainty_method[0]
            if len(uncertainty_method) > 1:
                self.uncertainty_params_list = [self.feature_dist_func]
                self.uncertainty_params_list += uncertainty_method[1:]
        
        self.exploitation_use_quantile_for_activity = exploitation_use_quantile_for_activity
        self.exploitation_sample_actives_from_clusters = exploitation_sample_actives_from_clusters
        self.exploitation_activity_threshold = exploitation_activity_threshold
        self.exploitation_use_quantile_for_weight = exploitation_use_quantile_for_weight
        self.exploitation_weight_threshold = exploitation_weight_threshold
        self.exploitation_alpha = exploitation_alpha
        self.exploitation_dissimilarity_lambda = exploitation_dissimilarity_lambda
        self.use_intra_cluster_threshold_for_exploitation = use_intra_cluster_threshold_for_exploitation
        self.use_proportional_cluster_budget_for_exploitation = use_proportional_cluster_budget_for_exploitation
        
        self.exploration_strategy = exploration_strategy
        self.exploration_use_quantile_for_weight = exploration_use_quantile_for_weight
        self.exploration_weight_threshold = exploration_weight_threshold
        self.exploration_beta = exploration_beta
        self.exploration_dissimilarity_lambda = exploration_dissimilarity_lambda
        self.use_intra_cluster_threshold_for_exploration = use_intra_cluster_threshold_for_exploration
        self.use_proportional_cluster_budget_for_exploration = use_proportional_cluster_budget_for_exploration
        
        # create pandas df for various cluster calculations
        # creates only cluster IDs for clusters with unlabeled instances 
        u_clusters, c_clusters = np.unique(self.clusters_unlabeled,
                                           return_counts=True)
        self.total_clusters = len(u_clusters)
        self.cluster_cols = ['Cluster ID', 'Cluster Mol Count',
                             'Density', 'Coverage', 
                             'Mean Uncertainty', 'Mean Activity Prediction',  'Mean Cost',
                             'High Activity Prediction Count',
                             'Exploitation Weight', 'Exploration Weight']
        self.clusters_df = pd.DataFrame(data=np.nan*np.zeros((self.total_clusters, len(self.cluster_cols))),
                                        columns=self.cluster_cols)
        self.clusters_df['Cluster ID'] = u_clusters
        self.clusters_df['Cluster Mol Count'] = c_clusters
        self.clusters_df.index = self.clusters_df['Cluster ID']
        
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
        uncertainty_unlabeled = self.trained_model.get_uncertainty(X=self.unlabeled_loader.get_features(), 
                                                                   uncertainty_method=self.uncertainty_method,
                                                                   uncertainty_params_list=self.uncertainty_params_list)
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            cluster_uncertainty = uncertainty_unlabeled[mol_idx]
            avg_cluster_uncertainty_i = np.nan_to_num(np.mean(cluster_uncertainty))
            self.clusters_df.loc[ci, 'Mean Uncertainty'] = avg_cluster_uncertainty_i
            
    def _compute_cluster_activity_prediction(self):
        preds_unlabeled = self.trained_model.predict(self.unlabeled_loader.get_features())[:,0] # get first task for now. TODO: account for multi-task setting?
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            cluster_preds = preds_unlabeled[mol_idx]
            if self.exploitation_use_quantile_for_activity:
                cluster_preds = cluster_preds[cluster_preds >= np.percentile(cluster_preds, 100.0*self.exploitation_activity_threshold)]
            else:
                cluster_preds = cluster_preds[cluster_preds >= self.exploitation_activity_threshold]
            avg_cluster_activity_i = np.nan_to_num(np.mean(cluster_preds))
            self.clusters_df.loc[ci, 'Mean Activity Prediction'] = avg_cluster_activity_i
            self.clusters_df.loc[ci, 'High Activity Prediction Count'] = len(cluster_preds)
            
    def _compute_cluster_cost(self):
        costs_unlabeled = self.unlabeled_loader.get_costs()
        for ci in self.clusters_df['Cluster ID']:
            mol_idx = np.where(self.clusters_unlabeled == ci)[0]
            avg_cluster_cost_i = np.nan_to_num(np.mean(costs_unlabeled[mol_idx]))
            self.clusters_df.loc[ci, 'Mean Cost'] = avg_cluster_cost_i
    
    def _compute_cluster_exploitation_weight(self):
        self.clusters_df['Exploitation Weight'] = (self.exploitation_alpha * self.clusters_df['Mean Activity Prediction']) + \
                                                  ((1 - self.exploitation_alpha) * self.clusters_df['Coverage'] * self.clusters_df['Density'])
    
    def _compute_cluster_exploration_weight(self):
        if self.exploration_strategy == "weighted":
            self.clusters_df['Exploration Weight'] = (self.exploration_beta * self.clusters_df['Mean Uncertainty']) + \
                                                     ((1 - self.exploration_beta) * (1 - self.clusters_df['Coverage']))
        elif self.exploration_strategy == "dissimilar":
            cluster_ids = self.clusters_df['Cluster ID'].values
            _, avg_cluster_dissimilarity = self._get_avg_cluster_dissimilarity(cluster_ids, 
                                                                               cluster_ids)
            self.clusters_df['Exploration Weight'] = avg_cluster_dissimilarity
            
    def _get_candidate_exploitation_clusters(self):
        if self.exploitation_use_quantile_for_weight:
            qualifying_exploitation_clusters = self.clusters_df['Exploitation Weight'] >= np.percentile(self.clusters_df['Exploitation Weight'], 100.0*self.exploitation_weight_threshold)
        else:
            qualifying_exploitation_clusters = self.clusters_df['Exploitation Weight'] >= self.exploitation_weight_threshold
        candidate_exploitation_clusters = self.clusters_df[qualifying_exploitation_clusters]
        
        # remove exploitation clusters that do not have positive high activity prediction
        if self.exploitation_sample_actives_from_clusters:
            candidate_exploitation_clusters = candidate_exploitation_clusters[candidate_exploitation_clusters['High Activity Prediction Count'] > 0]
        
        candidate_exploitation_clusters = candidate_exploitation_clusters['Cluster ID'].values
        return candidate_exploitation_clusters
        
    def _get_candidate_exploration_clusters(self):
        if self.exploration_strategy == "weighted":
            if self.exploration_use_quantile_for_weight:
                qualifying_exploration_clusters = self.clusters_df['Exploration Weight'] >= np.percentile(self.clusters_df['Exploration Weight'], 100.0*self.exploration_weight_threshold)
            else:
                qualifying_exploration_clusters = self.clusters_df['Exploration Weight'] >= self.exploration_weight_threshold
            candidate_exploration_clusters = self.clusters_df[qualifying_exploration_clusters]['Cluster ID'].values
        elif self.exploration_strategy == "random":
            candidate_exploration_clusters = self.clusters_df['Cluster ID'].values
        else: # self.exploration_strategy == "dissimilar"
            candidate_exploration_clusters = self.clusters_df['Cluster ID'].values
        return candidate_exploration_clusters
    
    def _get_candidate_exploitation_instances_total(self, cluster_ids):
        exploitation_instances_total = np.sum(self.clusters_df.loc[cluster_ids, 'High Activity Prediction Count'])
        return exploitation_instances_total
        
    """
        Currently not used. 
    """
    def _adjust_overlapping_clusters(self, candidate_exploitation_clusters, candidate_exploration_clusters):
        exploitation_clusters_to_drop = []
        exploration_clusters_to_drop = []
        overlapping_clusters = np.intersect1d(candidate_exploitation_clusters, candidate_exploration_clusters)
        for ci in overlapping_clusters:
            if self.clusters_df['Exploitation Weight'].loc[ci] >= self.clusters_df['Exploration Weight'].loc[ci]:
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
        selected_clusters_instances_pairs = None
        if useExploitationStrategy:
            selected_clusters_instances_pairs = self._select_instances_from_clusters_weighted(candidate_clusters, 
                                                                                              total_budget, 
                                                                                              self.exploitation_dissimilarity_lambda,
                                                                                              weight_column='Exploitation Weight',
                                                                                              useIntraClusterThreshold=self.use_intra_cluster_threshold_for_exploitation,
                                                                                              useProportionalClusterBudget=self.use_proportional_cluster_budget_for_exploitation,
                                                                                              selectDissimilarInstancesWithinCluster=self.select_dissimilar_instances_within_cluster)
        else:
            if self.exploration_strategy == "random":
                selected_clusters_instances_pairs = self._select_instances_from_clusters_random(candidate_clusters, 
                                                                                                total_budget,
                                                                                                useProportionalClusterBudget=self.use_proportional_cluster_budget_for_exploration,
                                                                                                selectDissimilarInstancesWithinCluster=self.select_dissimilar_instances_within_cluster)
            
            elif self.exploration_strategy == "dissimilar":
                selected_clusters_instances_pairs = self._select_instances_from_clusters_dissimilar(candidate_clusters, 
                                                                                                    total_budget,
                                                                                                    useProportionalClusterBudget=self.use_proportional_cluster_budget_for_exploration,
                                                                                                    selectDissimilarInstancesWithinCluster=self.select_dissimilar_instances_within_cluster)
            else:
                selected_clusters_instances_pairs = self._select_instances_from_clusters_weighted(candidate_clusters, 
                                                                                                  total_budget,
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
        if len(candidate_clusters) == 0:
            return selected_clusters_instances_pairs
        
        curr_cluster_budget = 0
        curr_cluster_budget = np.nan_to_num(np.ceil(total_budget / len(candidate_clusters)))
        if useProportionalClusterBudget:
            cluster_unlabeled_counts = self._get_candidate_exploration_instances_per_cluster_count(candidate_clusters)
            total_unlabeled_counts = np.sum(cluster_unlabeled_counts)
        
        if weight_column == 'Exploitation Weight':
            preds_unlabeled = self.trained_model.predict(self.unlabeled_loader.get_features())[:,0]
            
        remaining_total_budget = total_budget
        if remaining_total_budget > 0:
            # select highest weighted cluster first
            cluster_weights = self.clusters_df[weight_column].loc[candidate_clusters].values
            curr_selected_cluster_idx = np.argsort(cluster_weights)[::-1][0]
            curr_selected_cluster = candidate_clusters[curr_selected_cluster_idx]
            
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts))
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            cluster_instance_idx = np.where(self.clusters_unlabeled == curr_selected_cluster)[0]
            if weight_column == 'Exploitation Weight' and self.exploitation_sample_actives_from_clusters:
                cluster_preds = preds_unlabeled[cluster_instance_idx]
                if self.exploitation_use_quantile_for_activity:
                    cluster_instance_idx = cluster_instance_idx[cluster_preds >= np.percentile(cluster_preds, 100.0*self.exploitation_activity_threshold)]
                else:
                    cluster_instance_idx = cluster_instance_idx[cluster_preds >= self.exploitation_activity_threshold]
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances(cluster_instance_idx, 
                                                                                                         curr_cluster_budget,
                                                                                                         useIntraClusterThreshold=useIntraClusterThreshold)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances(cluster_instance_idx, 
                                                                                                     curr_cluster_budget)
            
            selected_clusters_instances_pairs.append((curr_selected_cluster,))
            selected_clusters_instances_pairs[-1] = selected_clusters_instances_pairs[-1] + (selected_instances_cluster,)
            remaining_total_budget -= len(selected_instances_cluster)
            
            rem_candidate_clusters =  np.ones_like(candidate_clusters).astype(bool)
            rem_candidate_clusters[curr_selected_cluster_idx] = False
            prev_sum_cluster_dissimilarity = np.zeros_like(candidate_clusters)
        
        # select remaining clusters based on what was already selected
        i=1
        while i < len(candidate_clusters) and remaining_total_budget > 0:
            start_time1 = time.time()
            last_selected_cluster = selected_clusters_instances_pairs[-1][0]
            _, avg_cluster_dissimilarity = self._get_avg_cluster_dissimilarity([last_selected_cluster], 
                                                                               candidate_clusters[rem_candidate_clusters])
            avg_cluster_dissimilarity = (avg_cluster_dissimilarity + prev_sum_cluster_dissimilarity[rem_candidate_clusters]) / len(selected_clusters_instances_pairs)
            
            cluster_weights = self.clusters_df[weight_column].loc[candidate_clusters[rem_candidate_clusters]].values
            # adjust cluster weights to include avg cluster dissimilarity
            adjusted_cluster_weights = dissimilarity_lambda * avg_cluster_dissimilarity + \
                                       ((1 - dissimilarity_lambda) * cluster_weights)
            
            highest_w_idx = np.argsort(adjusted_cluster_weights)[::-1][0]
            curr_selected_cluster = candidate_clusters[rem_candidate_clusters][highest_w_idx]
            curr_selected_cluster_idx = np.where(candidate_clusters == curr_selected_cluster)[0]
            rem_candidate_clusters[curr_selected_cluster_idx] = False
            
            # process current cluster budget
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts)) 
            curr_cluster_budget = curr_cluster_budget + remaining_cluster_budget
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            cluster_instance_idx = np.where(self.clusters_unlabeled == curr_selected_cluster)[0]
            if weight_column == 'Exploitation Weight' and self.exploitation_sample_actives_from_clusters:
                cluster_preds = preds_unlabeled[cluster_instance_idx]
                if self.exploitation_use_quantile_for_activity:
                    cluster_instance_idx = cluster_instance_idx[cluster_preds >= np.percentile(cluster_preds, 100.0*self.exploitation_activity_threshold)]
                else:
                    cluster_instance_idx = cluster_instance_idx[cluster_preds >= self.exploitation_activity_threshold]
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances(cluster_instance_idx, 
                                                                                                         curr_cluster_budget,
                                                                                                         useIntraClusterThreshold=useIntraClusterThreshold)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances(cluster_instance_idx, 
                                                                                                     curr_cluster_budget)
            
            prev_sum_cluster_dissimilarity[rem_candidate_clusters] = np.delete(avg_cluster_dissimilarity, highest_w_idx) * len(selected_clusters_instances_pairs)
            
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
        curr_cluster_budget = 0
        if len(candidate_clusters) != 0:
            curr_cluster_budget = np.ceil(total_budget / len(candidate_clusters))
        if useProportionalClusterBudget:
            cluster_unlabeled_counts = self._get_candidate_exploration_instances_per_cluster_count(candidate_clusters)
            total_unlabeled_counts = np.sum(cluster_unlabeled_counts)
        
        remaining_total_budget = total_budget
        i=0
        rem_clusters_idx = list(np.arange(len(candidate_clusters)))
        remaining_cluster_budget = 0
        while i < len(candidate_clusters) and remaining_total_budget > 0:
            curr_selected_cluster_idx = np.random.choice(rem_clusters_idx, size=1, replace=False)[0]
            rem_clusters_idx.remove(curr_selected_cluster_idx)
            curr_selected_cluster = candidate_clusters[curr_selected_cluster_idx]
            
            # process current cluster budget
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts)) 
            curr_cluster_budget = curr_cluster_budget + remaining_cluster_budget
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            cluster_instance_idx = np.where(self.clusters_unlabeled == curr_selected_cluster)[0]
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances(cluster_instance_idx, 
                                                                                                         curr_cluster_budget,
                                                                                                         useIntraClusterThreshold=False)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances(cluster_instance_idx, 
                                                                                                     curr_cluster_budget)
            selected_clusters_instances_pairs.append((curr_selected_cluster,))
            selected_clusters_instances_pairs[-1] = selected_clusters_instances_pairs[-1] + (selected_instances_cluster,)
            remaining_total_budget -= len(selected_instances_cluster)
            i+=1
            
        return selected_clusters_instances_pairs

    """
        Helper method for selecting clusters in a dissimilar manner.
    """
    def _select_instances_from_clusters_dissimilar(self,
                                                   candidate_clusters, 
                                                   total_budget,
                                                   useProportionalClusterBudget=False,
                                                   selectDissimilarInstancesWithinCluster=True):
        selected_clusters_instances_pairs = []
        curr_cluster_budget = 0
        if len(candidate_clusters) != 0:
            curr_cluster_budget = np.ceil(total_budget / len(candidate_clusters))
        if useProportionalClusterBudget:
            cluster_unlabeled_counts = self._get_candidate_exploration_instances_per_cluster_count(candidate_clusters)
            total_unlabeled_counts = np.sum(cluster_unlabeled_counts)
        
        remaining_total_budget = total_budget
        rem_clusters_idx = list(np.arange(len(candidate_clusters)))
        remaining_cluster_budget = 0
        # select first cluster randomly
        if remaining_total_budget > 0:
            curr_selected_cluster_idx = np.random.choice(rem_clusters_idx, size=1, replace=False)[0]
            rem_clusters_idx.remove(curr_selected_cluster_idx)
            curr_selected_cluster = candidate_clusters[curr_selected_cluster_idx]
            
            # process current cluster budget
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts)) 
            curr_cluster_budget = curr_cluster_budget + remaining_cluster_budget
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            cluster_instance_idx = np.where(self.clusters_unlabeled == curr_selected_cluster)[0]
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances(cluster_instance_idx, 
                                                                                                         curr_cluster_budget,
                                                                                                         useIntraClusterThreshold=False)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances(cluster_instance_idx, 
                                                                                                     curr_cluster_budget)
            selected_clusters_instances_pairs.append((curr_selected_cluster,))
            selected_clusters_instances_pairs[-1] = selected_clusters_instances_pairs[-1] + (selected_instances_cluster,)
            remaining_total_budget -= len(selected_instances_cluster)
            
        # select remaining clusters so that they are dissimilar to clusters already selected
        prev_sum_cluster_dissimilarity = np.zeros_like(candidate_clusters)
        rem_candidate_clusters = np.ones_like(candidate_clusters).astype(bool)
        i=1
        while i < len(candidate_clusters) and remaining_total_budget > 0:
            last_selected_cluster = selected_clusters_instances_pairs[-1][0]
            _, avg_cluster_dissimilarity = self._get_avg_cluster_dissimilarity([last_selected_cluster], 
                                                                               candidate_clusters[rem_candidate_clusters])
            avg_cluster_dissimilarity = (avg_cluster_dissimilarity + prev_sum_cluster_dissimilarity[rem_candidate_clusters]) / len(selected_clusters_instances_pairs)
            
            highest_w_idx = np.argsort(avg_cluster_dissimilarity)[::-1][0]
            curr_selected_cluster = candidate_clusters[rem_candidate_clusters][highest_w_idx]
            curr_selected_cluster_idx = np.where(candidate_clusters == curr_selected_cluster)[0]
            rem_candidate_clusters[curr_selected_cluster_idx] = False
            
            # process current cluster budget
            if useProportionalClusterBudget:
                curr_cluster_budget = np.ceil(total_budget * (cluster_unlabeled_counts[curr_selected_cluster_idx] /  total_unlabeled_counts)) 
            curr_cluster_budget = curr_cluster_budget + remaining_cluster_budget
            curr_cluster_budget = min(remaining_total_budget, curr_cluster_budget)
            
            cluster_instance_idx = np.where(self.clusters_unlabeled == curr_selected_cluster)[0]
            if selectDissimilarInstancesWithinCluster:
                selected_instances_cluster, remaining_cluster_budget = self._select_dissimilar_instances(cluster_instance_idx, 
                                                                                                         curr_cluster_budget,
                                                                                                         useIntraClusterThreshold=useIntraClusterThreshold)
            else:
                selected_instances_cluster, remaining_cluster_budget = self._select_random_instances(cluster_instance_idx, 
                                                                                                     curr_cluster_budget)
            
            prev_sum_cluster_dissimilarity[rem_candidate_clusters] = np.delete(avg_cluster_dissimilarity, highest_w_idx) * len(selected_clusters_instances_pairs)
            
            selected_clusters_instances_pairs.append((curr_selected_cluster,))
            selected_clusters_instances_pairs[-1] = selected_clusters_instances_pairs[-1] + (selected_instances_cluster,)
            remaining_total_budget -= len(selected_instances_cluster)
            
            i+=1
            
        return selected_clusters_instances_pairs
        
    def select_next_batch(self):
        start_time = time.time()
        # populate self.clusters_df
        self._compute_cluster_densities()
        self._compute_cluster_coverage()
        self._compute_cluster_uncertainty()
        self._compute_cluster_activity_prediction()
        self._compute_cluster_cost()
        
        # compute cluster exploitation and exploration weights
        self._compute_cluster_exploitation_weight()
        self._compute_cluster_exploration_weight()
        print("Done computing cluster properties. Took {} seconds.".format(time.time()-start_time))
        
        selected_exploitation_cluster_instances_pairs, selected_exploration_cluster_instances_pairs = super(ClusterBasedWCSelector, self).select_next_batch()
        
        # account for case when we have more room in the budget 
        # remaining budget allocated toward exploration by just picking the top-ranked unselected clusters by exploration weight
        selected_exploitation_clusters = []
        selected_exploitation_instances_count = 0
        selected_exploration_clusters = []
        selected_exploration_instances_count = 0
        if len(selected_exploitation_cluster_instances_pairs) > 0:
            selected_exploitation_clusters = [x[0] for x in selected_exploitation_cluster_instances_pairs]
            selected_exploitation_instances = [x[1] for x in selected_exploitation_cluster_instances_pairs]
            selected_exploitation_instances_count = np.hstack(selected_exploitation_instances).shape[0]
        
        if len(selected_exploration_cluster_instances_pairs) > 0:
            selected_exploration_clusters = [x[0] for x in selected_exploration_cluster_instances_pairs]
            selected_exploration_instances = [x[1] for x in selected_exploration_cluster_instances_pairs]
            selected_exploration_instances_count = np.hstack(selected_exploration_instances).shape[0]
        exploration_budget = self.batch_size - (selected_exploitation_instances_count + selected_exploration_instances_count)
        
        if exploration_budget > 0 and self.exploration_strategy == "weighted":
            candidate_exploration_clusters = np.setdiff1d(self.clusters_df['Cluster ID'].values, 
                                                          np.union1d(selected_exploitation_clusters, selected_exploration_clusters))
            
            candidate_exploration_clusters_weights = self.clusters_df['Exploration Weight'].loc[candidate_exploration_clusters].values
            top_k_exploration_clusters = np.argsort(candidate_exploration_clusters_weights)[::-1][:exploration_budget]
            candidate_exploration_clusters = candidate_exploration_clusters[top_k_exploration_clusters]
            
            remaining_selected_exploration_cluster_instances_pairs = self._select_instances_from_clusters(candidate_exploration_clusters, 
                                                                                                          exploration_budget, 
                                                                                                          useExploitationStrategy=False)
            selected_exploration_cluster_instances_pairs.extend(remaining_selected_exploration_cluster_instances_pairs)
            
        return (selected_exploitation_cluster_instances_pairs, 
                selected_exploration_cluster_instances_pairs)