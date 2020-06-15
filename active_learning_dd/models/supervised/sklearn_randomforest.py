"""
    Class wrapper for sklearn random forest models. Adds extra wrapper functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .supervised_model import SupervisedModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SklearnRF(SupervisedModel):
    def __init__(self, task_names,
                 n_estimators=100, 
                 max_features='auto',
                 min_samples_leaf=1,
                 n_jobs=1,
                 class_weight=None,
                 random_state=None,
                 oob_score=False,
                 verbose=1):
        super(SklearnRF, self).__init__(task_names)
        # model params
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.random_state = random_state
        self.oob_score = oob_score
        self.verbose = verbose
        
        # setup model
        self.model_dict = {}
        for task_name in self.task_names:
            self.model_dict[task_name] = RandomForestClassifier(n_estimators=self.n_estimators, 
                                                                max_features=self.max_features, 
                                                                min_samples_leaf=self.min_samples_leaf, 
                                                                n_jobs=self.n_jobs, 
                                                                class_weight=self.class_weight,
                                                                random_state=self.random_state,
                                                                oob_score=self.oob_score, 
                                                                verbose=self.verbose)
    
    def fit(self, X_train, y_train):
        if X_train.shape[0] > 0: # in case of no training data assume 0.5 prediction
            # perform random shuffling of training data (including X_train)
            np.random.seed(self.random_state)
            shuffle_p = np.random.permutation(X_train.shape[0])
            X_train = X_train[shuffle_p]
            y_train = y_train[shuffle_p]
            
            # fit model
            for ti, task_name in enumerate(self.task_names):
                y_train_ti = y_train[:,ti].flatten()
                nan_indices = np.where(np.isnan(y_train_ti))[0] # remove NaN labes
                y_train_ti = np.delete(y_train_ti, nan_indices, axis=0)
                X_task_ti = np.delete(X_train, nan_indices, axis=0)
                
                self.model_dict[task_name].fit(X_task_ti, y_train_ti)
                
            # set fit check
            self.check_fit = True
        
    def predict(self, X):
        y_pred = np.zeros(shape=(X.shape[0],len(self.task_names)))
        if self.check_fit:
            for ti, task_name in enumerate(self.task_names):
                task_preds = self.model_dict[task_name].predict_proba(X)
                # check if model only seen 1 class; i.e. if all samples are 0
                task_classes = self.model_dict[task_name].classes_
                if task_classes.shape[0] == 1:
                    y_pred[:,ti] = task_preds[:,0]
                    if task_classes[0] == 0: # flip if only seen class is 0; same as setting prob of class 1 as 0.0
                        y_pred[:,ti] = 1.0 - y_pred[:,ti]
                else:
                    y_pred[:,ti] = task_preds[:,1]
        else: # if model has not been fit, then assume 0.5 for all samples
            y_pred[:] = 0.5
        return y_pred
        
    """
        Uncertainty measured by QBC: query-by-committee.
        Uses kullback-Leibler divergence measure.
    """
    def get_uncertainty_qbc(self, X):
        uncertainty = np.zeros(shape=(X.shape[0],len(self.task_names)))
        for ti, task_name in enumerate(self.task_names):
            consensus_preds = self.model_dict[task_name].predict_proba(X)
            consensus_preds = np.clip(consensus_preds, a_min=1e-7, a_max=None)
            n_estimators = len(self.model_dict[task_name].estimators_)
            estimator_uncertainty_sum = np.zeros(shape=(X.shape[0],))
            for estimator in self.model_dict[task_name].estimators_:
                estimator_preds = estimator.predict_proba(X)
                estimator_preds = np.clip(estimator_preds, a_min=1e-7, a_max=None)
                kb_divs = estimator_preds * np.log(estimator_preds / consensus_preds)
                kb_divs = np.sum(kb_divs, axis=1)
                estimator_uncertainty_sum += kb_divs
            estimator_uncertainty_mean = estimator_uncertainty_sum / n_estimators
            uncertainty[:,ti] = estimator_uncertainty_mean
        return uncertainty