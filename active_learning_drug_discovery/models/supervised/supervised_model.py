"""
    Class wrapper for sklearn random forest models. Adds extra wrapper functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        if not isinstance(self.task_names, list):
            self.task_names = [self.task_names]
    
    @property
    def task_names(self):
        return self.task_names
        
    @task_name.setter
    def task_names(self, value):
        self.task_names = value
        if not isinstance(self.task_names, list):
            self.task_names = [self.task_names]
    
    def fit(self, X_train, y_train):
        return None
        
    def predict(self, X):
        return None