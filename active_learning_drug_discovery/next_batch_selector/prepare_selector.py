"""
    Helper script for loading next batch selector from config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .cluster_based_selector import *
 
def load_next_batch_selector(training_loader,
                             unlabeled_loader,
                             trained_model,
                             next_batch_selector_params):
    nbs_class = next_batch_selector_params["class"]
    return nbs_func_dict[nbs_class](training_loader,
                                    unlabeled_loader,
                                    trained_model,
                                    next_batch_selector_params)
    
def nbs_func_dict():
    return {"ClusterBasedWCSelector": prepare_ClusterBasedWCSelector,
            "InstanceBasedWCSelector": prepare_InstanceBasedWCSelector}

    
def prepare_ClusterBasedWCSelector(training_loader,
                                   unlabeled_loader,
                                   trained_model,
                                   next_batch_selector_params):
    CBWC_selector = ClusterBasedWCSelector(training_loader=training_loader,
                                           unlabeled_loader=unlabeled_loader,
                                           trained_model=trained_model,
                                           batch_size=next_batch_selector_params["batch_size"],
                                           intra_cluster_dissimilarity_threshold=next_batch_selector_params["intra_cluster_dissimilarity_threshold"],
                                           feature_dist_func=next_batch_selector_params["feature_dist_func",
                                           uncertainty_method=next_batch_selector_params["uncertainty_method"],
                                           select_dissimilar_instances_within_cluster=next_batch_selector_params["select_dissimilar_instances_within_cluster"],
                                           exploitation_activity_threshold=next_batch_selector_params["exploitation_activity_threshold"],
                                           exploitation_weight_threshold=next_batch_selector_params["exploitation_weight_threshold"],
                                           exploitation_alpha=next_batch_selector_params["exploitation_alpha"],
                                           exploitation_dissimilarity_lambda=next_batch_selector_params["exploitation_dissimilarity_lambda"],
                                           use_intra_cluster_threshold_for_exploitation=next_batch_selector_params["use_intra_cluster_threshold_for_exploitation"],
                                           use_proportional_cluster_budget_for_exploitation=next_batch_selector_params["use_proportional_cluster_budget_for_exploitation"],
                                           exploration_strategy=next_batch_selector_params["exploration_strategy"],
                                           exploration_weight_threshold=next_batch_selector_params["exploration_weight_threshold"],
                                           exploration_beta=next_batch_selector_params["exploration_beta"],
                                           exploration_dissimilarity_lambda=next_batch_selector_params["exploration_dissimilarity_lambda"],
                                           use_intra_cluster_threshold_for_exploration=next_batch_selector_params["use_intra_cluster_threshold_for_exploration"],
                                           use_proportional_cluster_budget_for_exploration=next_batch_selector_params["use_proportional_cluster_budget_for_exploration"])
    return CBWC_selector
    
    
def prepare_InstanceBasedWCSelector(training_loader,
                                    unlabeled_loader,
                                    trained_model,
                                    next_batch_selector_params):
    IBWC_selector = InstanceBasedWCSelector(training_loader=training_loader,
                                            unlabeled_loader=unlabeled_loader,
                                            trained_model=trained_model,
                                            batch_size=next_batch_selector_params["batch_size"],
                                            feature_dist_func=next_batch_selector_params["feature_dist_func",
                                            uncertainty_method=next_batch_selector_params["uncertainty_method"],
                                            exploitation_activity_threshold=next_batch_selector_params["exploitation_activity_threshold"],
                                            exploitation_weight_threshold=next_batch_selector_params["exploitation_weight_threshold"],
                                            exploitation_dissimilarity_lambda=next_batch_selector_params["exploitation_dissimilarity_lambda"],
                                            exploration_strategy=next_batch_selector_params["exploration_strategy"],
                                            exploration_weight_threshold=next_batch_selector_params["exploration_weight_threshold"],
                                            exploration_dissimilarity_lambda=next_batch_selector_params["exploration_dissimilarity_lambda"])
    return IBWC_selector