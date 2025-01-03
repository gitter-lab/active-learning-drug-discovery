"""
    Class wrapper for sklearn random forest models. Adds extra wrapper functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from active_learning_dd.utils.data_utils import get_avg_cluster_dissimilarity

class SupervisedModel(object):
    """Abstract base layer class.
    # Properties
        task_names: specifies the task names for the labels.
    # Methods
        fit(X_train, y_train): fits the model to the training data.
        predict(X): uses the current model to predict labels.
    """
    def __init__(self, task_names):
        self.task_names = task_names
        self.check_fit = False # check if model has been fit
    
    @property
    def task_names(self):
        return self._task_names
        
    @task_names.setter
    def task_names(self, value):
        self._task_names = value
        if not isinstance(self._task_names, list):
            self._task_names = [self._task_names]
    
    def fit(self, X_train, y_train):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
    """
        Uncertainty measured by least confidence/margin sampling.
    """
    def get_uncertainty_lc(self, X):
        preds = self.predict(X)
        #preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds)) # normalize in range 0-1.
        uncertainty = 1 - (np.abs(2*preds - 1))
        return uncertainty
    
    """
        Uncertainty measured by QBC: query-by-committee.
    """
    def get_uncertainty_qbc(self, X):
        raise NotImplementedError
    
    """
        Uncertainty measured by density-weight.
    """
    def get_uncertainty_dw(self, X,
                           feature_dist_func,
                           beta=1.0, 
                           useQBC=False):
        if useQBC:
            uncertainty = self.get_uncertainty_qbc(X)
        else:
            uncertainty = self.get_uncertainty_lc(X)
        
        clusters = np.arange(len(X))
        _, avg_dist = get_avg_cluster_dissimilarity(clusters, 
                                                    X, 
                                                    clusters, 
                                                    clusters,
                                                    feature_dist_func,
                                                    candidate_cluster_batch_size=100)
        avg_dist = 1.0 - avg_dist
        avg_dist = ((avg_dist)**beta)
        uncertainty = uncertainty * avg_dist
        return uncertainty
    
    """
        Wrapper method for getting uncertainty of instances.
    """
    def get_uncertainty(self, X, uncertainty_method,
                        uncertainty_params_list):
        if uncertainty_method == "least_confidence":
            uncertainty = self.get_uncertainty_lc(X)
        elif uncertainty_method == "query_by_committee":
            uncertainty = self.get_uncertainty_qbc(X)
        elif uncertainty_method == "density_weight":
            uncertainty = self.get_uncertainty_dw(X, 
                                                  feature_dist_func=uncertainty_params_list[0],
                                                  beta=uncertainty_params_list[1], 
                                                  useQBC=uncertainty_params_list[2])
        else:
            raise NotImplementedError
            
        return uncertainty