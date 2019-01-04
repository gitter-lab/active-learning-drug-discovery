"""
    Runs a single iteration of the active learning pipeline.
    
    Usage:
        python active_learning_dd.py \
        --params_json_file=../param_configs/params_set_0.json \
        --iter_num=$iter_num \ 
        --process_num=$process_num
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

from active_learning_dd.models.load_model import load_model
from active_learning_dd.database_loaders.load_data import prepare_loader


if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_params_json_file', action="store", dest="pipeline_params_json_file", required=True)
    parser.add_argument('--nbs_params_json_file', action="store", dest="params_json_file", required=True)
    parser.add_argument('--iter_num', type=int, default=0, action="store", dest="iter_num", required=True)
    parser.add_argument('--process_num', type=int, default=0, action="store", dest="process_num", required=True)
    
    given_args = parser.parse_args()
    pipeline_params_json_file = given_args.pipeline_params_json_file
    nbs_params_json_file = given_args.nbs_params_json_file
    iter_num = given_args.iter_num
    process_num = given_args.process_num
    
    # load param json configs
    with open(pipeline_params_json_file) as f:
        pipeline_config = json.load(f)
    with open(nbs_params_json_file) as f:
        nbs_config = json.load(f)
        
    # load training data
    training_loader = prepare_loader(data_loader_params=pipeline_config["training_data_params"], 
                                     task_names=pipeline_config["common"]["task_names"],
                                     smile_col_name=pipeline_config["common"]["smile_col_name"],
                                     feature_name=pipeline_config["common"]["feature_name"],
                                     cluster_col_name=pipeline_config["common"]["cluster_col_name"],
                                     molecule_id_col_name=pipeline_config["common"]["molecule_id_col_name"],
                                     cost_col_name=pipeline_config["common"]["cost_col_name"])
    X_train, y_train = training_loader.get_features_and_labels()
    
    # load and train model
    model = load_model(model_type=pipeline_config["model"]["type"]
                       model_class=pipeline_config["model"]["class"]
                       model_params=pipeline_config["model"]["params"]
                       task_names=pipeline_config["common"]["task_names"])
    model.fit(X_train, y_train)
    
    # load unlabeled pool
    unlabeled_loader = prepare_loader(data_loader_params=pipeline_config["unlabeled_data_params"], 
                                      task_names=pipeline_config["common"]["task_names"],
                                      smile_col_name=pipeline_config["common"]["smile_col_name"],
                                      feature_name=pipeline_config["common"]["feature_name"],
                                      cluster_col_name=pipeline_config["common"]["cluster_col_name"],,
                                      molecule_id_col_name=pipeline_config["common"]["molecule_id_col_name"],
                                      cost_col_name=pipeline_config["common"]["cost_col_name"])
    X_unlabeled = unlabeled_loader.get_features()
    
    # select next batch
    next_batch_selector = ClusterBasedSelector(training_loader=training_loader,
                                               unlabeled_loader=unlabeled_loader,
                                               trained_model=model,
                                               next_batch_selector_params=nbs_config["next_batch_selector_params"])