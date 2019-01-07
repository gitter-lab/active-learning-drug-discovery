"""
    Contains functions to load csv data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import os
import glob

class CSVLoader(object):
    """
    # Properties
        csv_file_or_dir: specifies a single file or path with format of csv files to be loaded. e.g: /path/iter_{}.csv or /path/iter_*.csv.
        task_names: specifies single or list of tasks/labels.
        smile_col_name: specifies the smiles column name.
        feature_name: specifies the feature name. Assumes some form of fingerprint features for now.
        cluster_col_name: specifies the cluster column name designating the cluster ids of each molecule.
    """
    def __init__(self, 
                 csv_file_or_dir,
                 task_names,
                 smile_col_name='rdkit SMILES',
                 feature_name='Morgan FP_2_1024',
                 cluster_col_name='Murcko Scaffold ID',
                 molecule_id_col_name='Molecule ID',
                 cost_col_name='Cost'):
        self.csv_file_or_dir = csv_file_or_dir
        self.task_names = task_names
        self.smile_col_name = smile_col_name
        self.cluster_col_name = cluster_col_name
        self.molecule_id_col_name = molecule_id_col_name
        
        if not isinstance(self.task_names, list):
            self.task_names = [self.task_names]
            
        # keep track of indices to drop for duplication purposes
        self.idx_to_drop = None
    
    @property
    def idx_to_drop(self):
        return self.idx_to_drop
        
    @idx_to_drop.setter
    def idx_to_drop(self, value):
        self.idx_to_drop = idx_to_drop
        if not isinstance(self.idx_to_drop, list):
            self.idx_to_drop = [self.idx_to_drop]
        
    def _load_dataframe(self):
        csv_files_list = [glob.glob(self.csv_file_or_dir.format('*'))]
        df_list = [pd.read_csv(csv_file) for csv_file in csv_files_list]
        data_df = pd.concat(df_list)
        if self.idx_to_drop is not None:
            data_df = data_df.drop(data_df.index[self.idx_to_drop])
        return data_df
        
    def get_dataframe(self):
        data_df = self._load_dataframe()
        return data_df
        
    def get_features(self):
        data_df = self.get_dataframe()
        X_data = data_df[self.feature_name].values
        X_data = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in X_data]).astype(float) # this is from: https://stackoverflow.com/a/29091970
        return X_data

    def get_labels(self):
        data_df = self.get_dataframe()
        y_data = data_df[self.task_names].values.astype(float)
        return y_data
        
    def get_features_and_labels(self):
        X_data = self.get_features()
        y_data = self.get_labels()
        return X_data, y_data
        
    def get_clusters(self):
        data_df = self.get_dataframe()
        return data_df[self.cluster_col_name].values.astype(float)
        
    def get_smiles(self):
        data_df = self.get_dataframe()
        return data_df[self.smile_col_name].values
        
    def get_molecule_ids(self):
        data_df = self.get_dataframe()
        return data_df[self.molecule_id_col_name].values
 
    def get_costs(self):
        data_df = self.get_dataframe()
        return data_df[self.cost_col_name].values.astype(float)