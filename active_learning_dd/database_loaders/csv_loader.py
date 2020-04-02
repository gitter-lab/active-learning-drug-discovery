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
from scipy import sparse

from ..utils.data_utils import get_duplicate_smiles_in1d

class CSVLoader(object):
    """
    # Properties
        csv_file_or_dir: specifies a single file or path with format of csv files to be loaded. e.g: /path/iter_{}.csv or /path/iter_*.csv.
        task_names: specifies single or list of tasks/labels.
        smile_col_name: specifies the smiles column name.
        feature_name: specifies the feature name. Assumes some form of fingerprint features for now.
        cluster_col_name: specifies the cluster column name designating the cluster ids of each molecule.
        idx_id_col_name: specifies the unique index ID for the compounds.
        cache_dataframes: if True, will maintain dataframes in memory. If False, will load it each time it is needed.
    """
    def __init__(self, 
                 csv_file_or_dir,
                 task_names,
                 smile_col_name='rdkit SMILES',
                 feature_name='Morgan FP_2_1024',
                 cluster_col_name='Murcko Scaffold ID',
                 molecule_id_col_name='Molecule',
                 cost_col_name='Cost',
                 idx_id_col_name='Index ID',
                 cache_dataframes=True,
                 convert_features_to_sparse_format=False):
        self.csv_file_or_dir = csv_file_or_dir
        self.task_names = task_names
        self.smile_col_name = smile_col_name
        self.feature_name = feature_name
        self.cluster_col_name = cluster_col_name
        self.molecule_id_col_name = molecule_id_col_name
        self.cost_col_name = cost_col_name
        self.idx_id_col_name = idx_id_col_name
        self.cache_dataframes = cache_dataframes
        self.convert_features_to_sparse_format = convert_features_to_sparse_format
        
        if self.cache_dataframes:
            self.data_df = None
            self.X_features = None
        
        self.num_files = len(glob.glob(self.csv_file_or_dir.format('*')))
        
        if not isinstance(self.task_names, list):
            self.task_names = [self.task_names]
            
        # keep track of indices to drop for duplication purposes
        self.idx_to_drop = None
    
    @property
    def idx_to_drop(self):
        return self._idx_to_drop
        
    @idx_to_drop.setter
    def idx_to_drop(self, value):
        self._idx_to_drop = value
        if self._idx_to_drop is not None and not isinstance(self._idx_to_drop, list):
            self._idx_to_drop = [self._idx_to_drop]
    
    def _load_dataframe(self):
        csv_files_list = glob.glob(self.csv_file_or_dir.format('*')) #[self.csv_file_or_dir.format(i) for i in range(self.num_files)]
        df_list = [pd.read_csv(csv_file) for csv_file in csv_files_list]
        data_df = pd.concat(df_list, sort=False)
        
        # remove duplicates via index id 
        if self.idx_id_col_name in data_df.columns:
            data_df = data_df.drop_duplicates(self.idx_id_col_name, keep="first")
        
        data_df = data_df.reset_index(drop=True)
        return data_df
        
    # remove already labeled molecules by checking other and unlabeled pool overlap
    # note duplicates determined via rdkit smiles
    def drop_duplicates_via_smiles(self, smiles_others):
        data_df = self._load_dataframe()
        smiles_this = data_df[self.smile_col_name].values
        idx_to_drop = get_duplicate_smiles_in1d(smiles_others, smiles_this)
        self.idx_to_drop = idx_to_drop
        
        # update cache after updating idx_to_drop
        if self.cache_dataframes:
            self.data_df = data_df
            if self.idx_to_drop is not None:
                self.data_df = self.data_df.drop(self.data_df.index.values[self.idx_to_drop])
        
    def get_dataframe(self):
        # if you are not caching or self.data_df is None, then load the dataframe from disk
        if (not self.cache_dataframes) or (self.data_df is None):
            data_df = self._load_dataframe()
            if self.idx_to_drop is not None:
                data_df = data_df.drop(data_df.index.values[self.idx_to_drop])
        # if you are caching, then just pass self.data_df if it is NOT None, 
        # otherwise set it to the loaded dataframe (since the above if should fire).
        if self.cache_dataframes:
            if self.data_df is None:
                self.data_df = data_df # pass by reference
            else:
                data_df = self.data_df # pass by reference
        return data_df
    
    def get_size(self):
        data_df = self.get_dataframe()
        return data_df.shape
    
    """
        Returns feature matrix for the desired rows.
        If desired rows is None, returns all rows; entire feature matrix.
    """
    def get_features(self, desired_rows=None):
        data_df = self.get_dataframe()
                
        if self.cache_dataframes:
            if self.X_features is None:
                X_features = data_df[self.feature_name].values
                X_features = np.vstack([ (np.fromstring(x, 'u1') - ord('0')).astype(np.uint16) for x in X_features ]) # this is from: https://stackoverflow.com/a/29091970
                self.X_features = X_features
            else:
                X_features = self.X_features
        else:
            X_features = data_df[self.feature_name].values
            X_features = np.vstack([ (np.fromstring(x, 'u1') - ord('0')).astype(np.uint16) for x in X_features ]) # this is from: https://stackoverflow.com/a/29091970
            
        if desired_rows is not None:
            X_features = X_features[desired_rows]
        
        if self.convert_features_to_sparse_format:
            X_features = sparse.coo_matrix(X_features)
            X_features = X_features.tocsc()
            
        return X_features

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
        return data_df[self.cluster_col_name].values
        
    def get_smiles(self):
        data_df = self.get_dataframe()
        return data_df[self.smile_col_name].values
        
    def get_molecule_ids(self):
        data_df = self.get_dataframe()
        return data_df[self.molecule_id_col_name].values
 
    def get_costs(self):
        data_df = self.get_dataframe()
        # if there is no cost column, then assume all costs are 1.0
        try:
            costs = data_df[self.cost_col_name].values.astype(float)
        except:
            costs = np.ones(shape=(data_df.shape[0],))
        return costs
        
    def get_idx_ids(self):
        data_df = self.get_dataframe()
        # if there is no index id column, then return range
        try:
            idx_ids = data_df[self.idx_id_col_name].values.astype('int64')
        except:
            idx_ids = np.arange(data_df.shape[0])
        return idx_ids