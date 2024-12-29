"""
    Script for generating the dissimilarity matrix.
    csv_file_or_dir: specifies a single file or path with format of csv files to be loaded. e.g: /path/iter_{}.csv or /path/iter_*.csv.
    output_dir: where to save the memmap file of the dissimilarity matrix.
    feature_name: specifies the column name for features in the csv file.
    cutoff: instances within this cutoff distance belong to the same cluster.
    dist_function: distance function to use.
    process: not used; can be ignored.
    
        Usage:
        python generate_dissimilarity_matrix.py \
        --csv_file_or_dir=../../datasets/lc_clusters_cv_96/unlabeled_{}.csv \
        --output_dir=../../datasets/ \
        --feature_name="Morgan FP_2_1024" \
        --dist_function="tanimoto_dissimilarity" \
        --process_count=4 \
        --process_batch_size=2056
"""
from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
import glob
import time
import pathlib
from multiprocessing import Process

from .data_utils import *

def get_features(csv_files_list, feature_name, index_name, tmp_dir, process_batch_size):
    # first get n_instances
    n_features = None
    instances_per_file = []
    for f in csv_files_list:
        for chunk in pd.read_csv(f, chunksize=process_batch_size):
            instances_per_file.append(chunk.shape[0])
            if not n_features:
                n_features = len(chunk[feature_name].iloc[0])
   
    n_instances = np.sum(instances_per_file)
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='w+', shape=(n_instances, n_features))
    chunksize = process_batch_size
    for i, f in enumerate(csv_files_list):
        for chunk in pd.read_csv(f, chunksize=chunksize):
            for batch_i in range(instances_per_file[i]//chunksize + 1): 
                row_start = batch_i*chunksize
                row_end = min(instances_per_file[i], (batch_i+1)*chunksize)
                if i > 0:
                    row_start = np.sum(instances_per_file[:i]) + batch_i*chunksize
                    row_end = min(np.sum(instances_per_file[:(i+1)]), np.sum(instances_per_file[:i]) + (batch_i+1)*chunksize)
                
                X[chunk[index_name].values.astype('int64'),:] = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in chunk[feature_name]]).astype(float) # this is from: https://stackoverflow.com/a/29091970
    X.flush()
    return n_instances, n_features
        
"""
    Function wrapper method for computing dissimilarity_matrix for a range of indices.
    Used with multiprocessing.
"""
def compute_dissimilarity_matrix_wrapper(start_ind, end_ind,
                                         n_instances, n_features,
                                         tmp_dir, output_dir, dist_func,
                                         process_id, process_batch_size):
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='r', shape=(n_instances, n_features))
    dissimilarity_matrix = np.memmap(output_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                     dtype='float16', mode='r+', shape=(n_instances, n_instances))
    dissimilarity_process_matrix = np.load(tmp_dir+'/dissimilarity_process_matrix.npy')[start_ind:end_ind]
    
    for i in range(end_ind-start_ind):
        start_time = time.time()
        row_start, row_end, col_start, col_end = dissimilarity_process_matrix[i,:]
        X_cols = X[col_start:col_end] 
        X_rows = X[row_start:row_end]
        dist_col_row = dist_func(X_cols, X_rows, X_batch_size=process_batch_size//2, Y_batch_size=process_batch_size//2)
        dist_col_row = dist_col_row.reshape(X_cols.shape[0], X_rows.shape[0])
            
        dissimilarity_matrix[row_start:row_end, col_start:col_end] = dist_col_row.T
        dissimilarity_matrix[col_start:col_end, row_start:row_end] = dist_col_row
        end_time = time.time()
        print('pid: {}, at {} of {}. time {} seconds.'.format(process_id, i, (end_ind-start_ind), (end_time-start_time)))
    del dissimilarity_matrix
 
def compute_dissimilarity_matrix(csv_file_or_dir, output_dir, feature_name='Morgan FP_2_1024', dist_function='tanimoto_dissimilarity', 
                                 process_count=1, process_batch_size=2056, index_name='Index ID'):
    num_files = len(glob.glob(csv_file_or_dir.format('*')))
    csv_files_list = [csv_file_or_dir.format(i) for i in range(num_files)]
    df_list = [pd.read_csv(csv_file) for csv_file in csv_files_list]
    data_df = pd.concat(df_list)
    
    # create tmp directory to store memmap arrays
    tmp_dir = './tmp/'
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
    n_instances, n_features = get_features(csv_files_list, feature_name, index_name, tmp_dir, process_batch_size)
    dist_func = feature_dist_func_dict()[dist_function]
    
    # compute_dissimilarity_matrix
    print('Generating dissimilarity_matrix...')
    start_time = time.time()
    dissimilarity_matrix = np.memmap(output_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                     dtype='float16', mode='w+', shape=(n_instances, n_instances))
    del dissimilarity_matrix
    
    # precompute indices of slices for dissimilarity_matrix
    examples_per_slice = n_instances//process_count
    dissimilarity_process_matrix = []
    row_batch_size = process_batch_size // 2
    col_batch_size = process_batch_size // 2
    num_slices = 0
    for process_id in range(process_count): 
        start_ind = process_id*examples_per_slice
        end_ind = (process_id+1)*examples_per_slice
        if process_id == (process_count-1):
            end_ind = n_instances
        if start_ind >= n_instances:
            break
        num_cols = end_ind - start_ind
        for batch_col_i in range(num_cols//col_batch_size + 1):
            col_start = start_ind + batch_col_i*col_batch_size
            col_end = min(end_ind, start_ind + (batch_col_i+1)*col_batch_size)
            for batch_row_i in range(col_end//row_batch_size + 1):
                row_start = batch_row_i*row_batch_size
                row_end = min(col_end, (batch_row_i+1)*row_batch_size)
                dissimilarity_process_matrix.append([row_start, row_end, col_start, col_end])
                num_slices += 1
    dissimilarity_process_matrix = np.array(dissimilarity_process_matrix)
    np.save(tmp_dir+'/dissimilarity_process_matrix.npy', dissimilarity_process_matrix)
    del dissimilarity_process_matrix
    print(num_slices)
    
    # distribute slices among processes
    process_pool = []
    slices_per_process = num_slices//process_count
    for process_id in range(process_count): 
        start_ind = process_id*slices_per_process
        end_ind = (process_id+1)*slices_per_process
        if process_id == (process_count-1):
            end_ind = num_slices
            
        if start_ind >= num_slices:
            break
            
        process_pool.append(Process(target=compute_dissimilarity_matrix_wrapper, args=(start_ind, end_ind,
                                                                                       n_instances, n_features,
                                                                                       tmp_dir, output_dir, dist_func,
                                                                                       process_id, process_batch_size)))
        process_pool[process_id].start()
    for process in process_pool:
        process.join()
        process.terminate()
        
    end_time = time.time()
    total_time = (end_time-start_time)/3600.0
    print('Done generating dissimilarity_matrix. Took {} hours'.format(total_time))
    
    import shutil
    shutil.rmtree(tmp_dir)
 
np.random.seed(1103)
if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_or_dir', action="store", dest="csv_file_or_dir", required=True)
    parser.add_argument('--output_dir', action="store", dest="output_dir", required=True)
    parser.add_argument('--feature_name', default='Morgan FP_2_1024', action="store", 
                        dest="feature_name", required=False)
    parser.add_argument('--dist_function', default='tanimoto_dissimilarity', action="store", 
                        dest="dist_function", required=False)
    parser.add_argument('--process_count', type=int, default=1, action="store", dest="process_count", required=False)
    parser.add_argument('--process_batch_size', type=int, default=2**17, action="store", dest="process_batch_size", required=False)
    parser.add_argument('--index_name', default='Index ID', action="store", dest="index_name", required=False)
    
    given_args = parser.parse_args()
    csv_file_or_dir = given_args.csv_file_or_dir
    output_dir = given_args.output_dir
    feature_name = given_args.feature_name
    dist_function = given_args.dist_function
    process_count = given_args.process_count
    process_batch_size = given_args.process_batch_size
    index_name = given_args.index_name
    
    compute_dissimilarity_matrix(csv_file_or_dir, output_dir, feature_name, dist_function, 
                                 process_count, process_batch_size, index_name)