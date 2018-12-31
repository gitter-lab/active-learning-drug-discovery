"""
    Helper script for loading data from config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from . import *
 
def prepare_loader(data_loader_params, 
                   task_names,
                   smile_col_name,
                   feature_name,
                   cluster_col_name):
    data_loader = data_loader_params["data_loader"]
    return data_loader_dict[data_loader](data_loader_params, 
                                         task_names,
                                         smile_col_name,
                                         feature_name,
                                         cluster_col_name)
    
def data_loader_dict():
    return {"CSVLoader": prepare_CSVLoader}
    
def prepare_CSVLoader(data_loader_params, 
                      task_names,
                      smile_col_name,
                      feature_name, 
                      cluster_col_name):
    training_csvloader = CSVLoader(csv_file_or_dir=data_loader_params["data_path_format"],
                                   task_names=task_names,
                                   smile_col_name=smile_col_name,
                                   feature_name=feature_name,
                                   cluster_col_name=cluster_col_name)
    return training_csvloader