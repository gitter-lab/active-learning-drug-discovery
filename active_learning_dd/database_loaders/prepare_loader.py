"""
    Helper script for loading data from config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .csv_loader import *
 
def prepare_loader(data_loader_params, 
                   task_names):
    data_loader_class = data_loader_params["data_loader_class"]
    return data_loader_dict()[data_loader_class](data_loader_params,
                                                 task_names)
    
def data_loader_dict():
    return {"CSVLoader": prepare_CSVLoader}
    
def prepare_CSVLoader(data_loader_params, 
                      task_names):
    training_csvloader = CSVLoader(csv_file_or_dir=data_loader_params["data_path_format"],
                                   task_names=task_names,
                                   smile_col_name=data_loader_params["smile_col_name"],
                                   feature_name=data_loader_params["feature_name"],
                                   cluster_col_name=data_loader_params["cluster_col_name"],
                                   molecule_id_col_name=data_loader_params["molecule_id_col_name"],
                                   cost_col_name=data_loader_params["cost_col_name"])
    return training_csvloader