"""
    Contains classes for instance based selectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cluster_based_selector import ClusterBasedWCSelector
import pandas as pd
import numpy as np


                
class InstanceBasedWCSelector(ClusterBasedWCSelector):
    """
    Selects next batch based on instance information.
    Weighted-Instances (WC): Uses a exploitation-exploration weighting scheme to select among instance.
                             Can explore exploration istances by random or dissimilar selection from unlabeled instances.
    Note this class inherits ClusterBasedWCSelector by just assigning each instance its own unique cluster.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 batch_size=384,
                 intra_cluster_dissimilarity_threshold=0.0,
                 feature_dist_func="tanimoto_dissimilarity",
                 uncertainty_method="least_confidence",
                 exploitation_use_quantile_for_activity=False,
                 exploitation_sample_actives_from_clusters=False,
                 exploitation_activity_threshold=0.75,
                 exploitation_use_quantile_for_weight=False,
                 exploitation_weight_threshold=0.5,
                 exploitation_dissimilarity_lambda=0.5,
                 exploration_strategy="weighted",
                 exploration_use_quantile_for_weight=False,
                 exploration_weight_threshold=0.5,
                 exploration_dissimilarity_lambda=0.5):
        super(InstanceBasedWCSelector, self).__init__(training_loader,
                                                      unlabeled_loader,
                                                      trained_model,
                                                      batch_size=batch_size,
                                                      intra_cluster_dissimilarity_threshold=0.0,
                                                      feature_dist_func=feature_dist_func,
                                                      uncertainty_method=uncertainty_method,
                                                      select_dissimilar_instances_within_cluster=False,
                                                      exploitation_use_quantile_for_activity=exploitation_use_quantile_for_activity,
                                                      exploitation_sample_actives_from_clusters=exploitation_sample_actives_from_clusters,
                                                      exploitation_activity_threshold=exploitation_activity_threshold,
                                                      exploitation_use_quantile_for_weight=exploitation_use_quantile_for_weight,
                                                      exploitation_weight_threshold=exploitation_weight_threshold,
                                                      exploitation_alpha=1.0,
                                                      exploitation_dissimilarity_lambda=exploitation_dissimilarity_lambda,
                                                      use_intra_cluster_threshold_for_exploitation=False,
                                                      use_proportional_cluster_budget_for_exploitation=False,
                                                      exploration_strategy=exploration_strategy,
                                                      exploration_use_quantile_for_weight=exploration_use_quantile_for_weight,
                                                      exploration_weight_threshold=0.5,
                                                      exploration_beta=1.0,
                                                      exploration_dissimilarity_lambda=exploration_dissimilarity_lambda,
                                                      use_intra_cluster_threshold_for_exploration=False,
                                                      use_proportional_cluster_budget_for_exploration=False)
        # assign each instance to unique cluster
        self.clusters_train = np.arange(len(self.clusters_train))
        self.clusters_unlabeled = np.arange(len(self.clusters_train), 
                                            len(self.clusters_train) + len(self.clusters_unlabeled))
        
        # redefine pandas df for instance case
        self.total_clusters = len(self.clusters_train) + len(self.clusters_unlabeled)
        self.clusters_df = pd.DataFrame(data=np.nan*np.zeros((self.total_clusters, len(self.cluster_cols)),
                                        columns=self.cluster_cols)
        self.clusters_df['Cluster ID'] = np.arange(len(self.clusters_train) + len(self.clusters_unlabeled))
        self.clusters_df['Cluster Mol Count'] = 1
        self.clusters_df.index = self.clusters_df['Cluster ID']
        
    def select_next_batch(self):
        return super(InstanceBasedWCSelector, self).select_next_batch()