"""
    Helper script for loading model from config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .supervised import *
from .unsupervised import *
 
def prepare_model(model_params,
                  task_names):
    model_type = model_params["type"]
    model_class = model_params["class"]
    model_params = model_params["params"]
    return model_func_dict()[model_type][model_class](model_params, task_names)
    
def model_func_dict():
    return {"supervised": {"sklearn_randomforest": prepare_rf_model,
                           "simple_nn": prepare_simple_nn},
            "unsupervised": {}}

    
def prepare_rf_model(rf_params, task_names):
    rf_model = sklearn_randomforest.SklearnRF(task_names,
                                             rf_params['n_estimators'],
                                             rf_params['max_features'],
                                             rf_params['min_samples_leaf'],
                                             rf_params['n_jobs'],
                                             rf_params['class_weight'],
                                             rf_params['random_state'],
                                             rf_params['oob_score'],
                                             rf_params['verbose'])
    return rf_model
    
def prepare_simple_nn(nn_params, task_names):
    nn_model = neural_network.SimpleNN(task_names,
                                       nn_params['n_features'],
                                       nn_params['batch_size'],
                                       nn_params['epochs'],
                                       nn_params['verbose'])
    return nn_model