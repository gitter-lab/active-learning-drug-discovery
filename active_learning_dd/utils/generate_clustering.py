"""
    Script for generating clusters using distance function. 
    NOTE: this uses rdkit implementation of Taylor-Butina clustering (unoptimizied for large datasets). See generate_bt_clustering for larger datasets.
    
    csv_file_or_dir: specifies a single file or path with format of csv files to be loaded. e.g: /path/iter_{}.csv or /path/iter_*.csv.
    output_dir: where to save the modified input csv files with cluster information added.
    feature_name: specifies the column name for features in the csv file.
    cutoff: instances within this cutoff distance belong to the same cluster.
    dist_function: distance function to use.
    process: not used; can be ignored.
    
        Usage:
        python generate_clustering.py \
        --csv_file_or_dir=../datasets/file_{}.csv \
        --output_dir=../datasets/ \
        --feature_name="Morgan FP_2_1024" \
        --cutoff=0.3 \
        --dist_function=tanimoto_dissimilarity \
        --process=$process
"""
from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import glob

from data_utils import *

def get_features(X_data):
    X_data = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in X_data]).astype(float) # this is from: https://stackoverflow.com/a/29091970
    return X_data
        
def cluster_features(X, dist_func, cutoff=0.2):
    # first generate the distance matrix
    n_instances = len(X)
    dists = np.zeros(shape=(n_instances*(n_instances-1)//2,), dtype='float16')
    dists = list(dists)
    nSoFar=0 
    for col in range(1, n_instances): 
        for row in range(col): 
            X_col = X[col] 
            X_row = X[row] 
            dist_col_row = dist_func(X_col, X_row) 
            dists[nSoFar] = dist_col_row 
            nSoFar += 1

    # now cluster the data:
    clusters = Butina.ClusterData(dists, n_instances, 
                                  cutoff, isDistData=True)
    return clusters
    
np.random.seed(1103)
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_or_dir', action="store", dest="csv_file_or_dir", required=True)
    parser.add_argument('--output_dir', action="store", dest="output_dir", required=True)
    parser.add_argument('--feature_name', default='Morgan FP_2_1024', action="store", 
                        dest="feature_name", required=False)
    parser.add_argument('--cutoff', type=float, default=0.2, action="store", dest="cutoff", required=False)
    parser.add_argument('--dist_function', default='tanimoto_dissimilarity', action="store", 
                        dest="dist_function", required=False)
    parser.add_argument('--process', type=int, default=0, action="store", dest="process", required=False)
    
    given_args = parser.parse_args()
    csv_file_or_dir = given_args.csv_file_or_dir
    output_dir = given_args.output_dir
    feature_name = given_args.feature_name
    cutoff = given_args.cutoff
    dist_function = given_args.dist_function
    process = given_args.process

    csv_files_list = glob.glob(csv_file_or_dir.format('*'))
    df_list = [pd.read_csv(csv_file) for csv_file in csv_files_list]
    data_df = pd.concat(df_list)
    X = get_features(data_df[feature_name].values) 
    dist_func = feature_dist_func_dict()[dist_function]
    
    # cluster
    clusters = cluster_features(X, dist_func, cutoff)
    
    # save resulting cluster ids
    data_df_cluster_ids = np.zeros(shape=(X.shape[0],), dtype=np.uint)
    for cluster_idx, cluster_tuple in enumerate(clusters):
        for instance_idx in cluster_tuple:
            data_df_cluster_ids[instance_idx] = cluster_idx
                
    data_df['Cluster_{}'.format(cutoff)] = data_df_cluster_ids
    
    start = 0
    for i in range(len(csv_files_list)):
        output_file_postfix = csv_files_list[i].split('/')[-1]
        curr_size = df_list[i].shape[0]
        end = start + curr_size
        data_df_slice_df = data_df.iloc[start:end,:]
        start = end
        data_df_slice_df.to_csv(output_dir+'/{}'.format(output_file_postfix))