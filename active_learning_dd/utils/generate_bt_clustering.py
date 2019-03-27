"""
    Script for generating clusters using Butina-Taylor esque method with a specified distance function.
    csv_file_or_dir: specifies a single file or path with format of csv files to be loaded. e.g: /path/iter_{}.csv or /path/iter_*.csv.
    output_dir: where to save the modified input csv files with cluster information added.
    feature_name: specifies the column name for features in the csv file.
    cutoff: instances within this cutoff distance belong to the same cluster.
    dist_function: distance function to use.
    process_count: number of processes to use when computing near neighbors.
    
        Usage:
        python generate_bt_clustering.py \
        --csv_file_or_dir=../../datasets/file_*.csv \
        --output_dir=../../datasets/ \
        --feature_name="Morgan FP_2_1024" \
        --cutoff=0.3 \
        --dist_function=tanimoto_dissimilarity \
        --process_count=4 \
        --dissimilarity_memmap_filename=../../datasets/dissimilarity_matrix_94857_94857.dat
"""
from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
import glob
from multiprocessing import Process
import pathlib
import time

from data_utils import *
    
def get_features(csv_files_list, feature_name, index_name, tmp_dir, process_batch_size) :
    # first get n_instances
    instances_per_file = []
    for f in csv_files_list:
        for chunk in pd.read_csv(f, chunksize=process_batch_size):
            instances_per_file.append(chunk.shape[0])
            
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
def compute_dissimilarity_matrix_wrapper_oldest(start_ind, end_ind,
                                                n_instances, n_features,
                                                tmp_dir, dist_func,
                                                process_id, process_batch_size):
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='r', shape=(n_instances, n_features))
    dissimilarity_matrix = np.memmap(tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                     dtype='float16', mode='r+', shape=(n_instances, n_instances))
    row_batch_size = process_batch_size
    for col in range(start_ind, end_ind): 
        start_time = time.time()
        for batch_i in range(col//row_batch_size + 1):
            row_start = batch_i*row_batch_size
            row_end = min(col, (batch_i+1)*row_batch_size)
            X_col = X[col] 
            X_row = X[row_start:row_end] 
            dist_col_row = dist_func(X_col, X_row, X_batch_size=50, Y_batch_size=row_batch_size)
            dissimilarity_matrix[row_start:row_end, col] = dist_col_row 
            dissimilarity_matrix[col, row_start:row_end] = dist_col_row 
        
        end_time = time.time()
        print('pid: {}, at {} with start {} end {}. time {} seconds.'.format(process_id, col, start_ind, end_ind, (end_time-start_time)))
    del dissimilarity_matrix

"""
    Function wrapper method for computing dissimilarity_matrix for a range of indices.
    Used with multiprocessing.
"""
def compute_dissimilarity_matrix_wrapper_old(start_ind, end_ind,
                                             n_instances, n_features,
                                             tmp_dir, dist_func,
                                             process_id, process_batch_size):
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='r', shape=(n_instances, n_features))
    dissimilarity_matrix = np.memmap(tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                     dtype='float16', mode='r+', shape=(n_instances, n_instances))
    row_batch_size = process_batch_size
    col_batch_size = process_batch_size // 4
    num_cols = end_ind - start_ind
    for batch_col_i in range(num_cols//col_batch_size + 1):
        col_start = start_ind + batch_col_i*col_batch_size
        col_end = min(end_ind, start_ind + (batch_col_i+1)*col_batch_size)
        X_cols = X[col_start:col_end] 
        start_time = time.time()
        for batch_row_i in range(col_end//row_batch_size + 1):
            row_start = batch_row_i*row_batch_size
            row_end = min(col_end, (batch_row_i+1)*row_batch_size)
            X_rows = X[row_start:row_end] 
            dist_col_row = dist_func(X_cols, X_rows, X_batch_size=process_batch_size//2, Y_batch_size=process_batch_size//2)
            dist_col_row = dist_col_row.reshape(X_cols.shape[0], X_rows.shape[0])
            
            dissimilarity_matrix[row_start:row_end, col_start:col_end] = dist_col_row.T
            dissimilarity_matrix[col_start:col_end, row_start:row_end] = dist_col_row
        end_time = time.time()
        print('pid: {}, at {} of {}. time {} seconds.'.format(process_id, batch_col_i, num_cols//col_batch_size + 1, (end_time-start_time)))
    del dissimilarity_matrix
   
"""
    Function wrapper method for computing dissimilarity_matrix for a range of indices.
    Used with multiprocessing.
"""
def compute_dissimilarity_matrix_wrapper(start_ind, end_ind,
                                         n_instances, n_features,
                                         tmp_dir, dist_func,
                                         process_id, process_batch_size,
                                         dissimilarity_memmap_filename):
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='r', shape=(n_instances, n_features))
    if dissimilarity_memmap_filename is None:
        dissimilarity_memmap_filename = tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances)
    dissimilarity_matrix = np.memmap(dissimilarity_memmap_filename, 
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
    
"""
    Function wrapper method for computing total number of nearest neigbors within cutoff for a set of instances.
    Used with multiprocessing.
"""
def compute_nn_total_wrapper(start_ind, end_ind,
                             n_instances, cutoff,
                             tmp_dir, 
                             process_id, process_batch_size,
                             dissimilarity_memmap_filename):
    if dissimilarity_memmap_filename is None:
        dissimilarity_memmap_filename = tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances)
        
    dissimilarity_matrix = np.memmap(dissimilarity_memmap_filename, 
                                     dtype='float16', mode='r', shape=(n_instances, n_instances))
    nn_total_vector = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(cutoff), 
                                dtype='int32', mode='r+', shape=(n_instances,))
    neighbor_matrix = np.memmap(tmp_dir+'/neighbor_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                dtype='uint8', mode='r+', shape=(n_instances, n_instances))
    row_batch_size = process_batch_size
    n_cols = n_instances
    for row in range(start_ind, end_ind): 
        row_total_neighbors = 0
        for batch_i in range(n_cols//row_batch_size + 1): 
            col_start = batch_i*row_batch_size
            col_end = (batch_i+1)*row_batch_size
            dm_slice = dissimilarity_matrix[row, col_start:col_end]
            row_total_neighbors += np.sum(dm_slice <= cutoff)
            
            neighbor_idxs = col_start + np.where(dm_slice <= cutoff)[0]
            neighbor_matrix[row, neighbor_idxs] = 1
            neighbor_matrix[neighbor_idxs, row] = 1
        nn_total_vector[row] = row_total_neighbors
    del nn_total_vector
    del neighbor_matrix

"""
    Function wrapper method for clustering singletons.
    Used with multiprocessing.
"""
def cluster_singletons_wrapper(start_ind, end_ind,
                               n_instances, cutoff,
                               output_dir, tmp_dir, 
                               cluster_id_start, 
                               process_id, process_batch_size,
                               dissimilarity_memmap_filename):
    if dissimilarity_memmap_filename is None:
        dissimilarity_memmap_filename = tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances)
        
    dissimilarity_matrix = np.memmap(dissimilarity_memmap_filename, 
                                     dtype='float16', mode='r', shape=(n_instances, n_instances))
    nn_total_vector = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(cutoff), 
                                dtype='int32', mode='r', shape=(n_instances,))
    cluster_assigment_vector = np.memmap(output_dir+'/cluster_assigment_vector_{}.dat'.format(cutoff), 
                                         dtype='int32', mode='r+', shape=(n_instances,))
    cluster_leader_idx_vector = np.memmap(tmp_dir+'/cluster_leader_idx_vector_{}.dat'.format(cutoff), 
                                          dtype='int32', mode='r+', shape=(n_instances,))
    row_batch_size = process_batch_size
    n_cols = n_instances
    for row in range(start_ind, end_ind): 
        if nn_total_vector[row] == 1:
            row_total_neighbors = 0
            closest_neighbor_idx = row
            closest_neighbor_dist = 1.0
            for batch_i in range(n_cols//row_batch_size + 1): 
                col_start = batch_i*row_batch_size
                col_end = (batch_i+1)*row_batch_size
                if col_start <= row and row < col_end:
                    dm_slice_1 = dissimilarity_matrix[row, col_start:row]
                    dm_slice_2 = dissimilarity_matrix[row, (row+1):col_end]
                    
                    # process dm_slice_1
                    if dm_slice_1.shape[0] > 0:
                        min_idx_slice = np.argmin(dm_slice_1)
                        min_dist = dm_slice_1[min_idx_slice]
                        if min_dist <= cutoff:
                            leader_idx = cluster_leader_idx_vector[col_start + min_idx_slice]
                            cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                            if cluster_leader_dist < closest_neighbor_dist:
                                closest_neighbor_idx = leader_idx
                                closest_neighbor_dist = cluster_leader_dist
                    
                    # process dm_slice_2
                    if dm_slice_2.shape[0] > 0:
                        min_idx_slice = np.argmin(dm_slice_2)
                        min_dist = dm_slice_2[min_idx_slice]
                        if min_dist <= cutoff:
                            leader_idx = cluster_leader_idx_vector[(row+1) + min_idx_slice]
                            cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                            if cluster_leader_dist < closest_neighbor_dist:
                                closest_neighbor_idx = leader_idx
                                closest_neighbor_dist = cluster_leader_dist
                else:
                    dm_slice = dissimilarity_matrix[row, col_start:col_end]
                    min_idx_slice = np.argmin(dm_slice)
                    min_dist = dm_slice[min_idx_slice]
                    if min_dist <= cutoff:
                        leader_idx = cluster_leader_idx_vector[col_start + min_idx_slice]
                        cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                        if cluster_leader_dist < closest_neighbor_dist:
                            closest_neighbor_idx = leader_idx
                            closest_neighbor_dist = cluster_leader_dist
            
            if closest_neighbor_idx == row: # true singleton case -> assign to unique cluster
                cluster_assigment_vector[row] = cluster_id_start + row
                cluster_leader_idx_vector[row] = row
                cluster_assigment_vector.flush()
                cluster_leader_idx_vector.flush()
            else: # false singleton case -> assign to cluster of closest cluster leader
                cluster_assigment_vector[row] = cluster_assigment_vector[closest_neighbor_idx]
                cluster_leader_idx_vector[row] = cluster_leader_idx_vector[closest_neighbor_idx]
                cluster_assigment_vector.flush()
                cluster_leader_idx_vector.flush()
    del cluster_assigment_vector
    del cluster_leader_idx_vector
    
def cluster_features(n_instances, n_features, dist_func, output_dir, tmp_dir, 
                     cutoff=0.2, process_count=1, process_batch_size=2**17,
                     dissimilarity_memmap_filename=None):
    total_clustering_time = 0
    # step 1: generate 
    print('Generating dissimilarity_matrix...')
    start_time = time.time()
    if dissimilarity_memmap_filename is None:
        dissimilarity_matrix = np.memmap(tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
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
    
    if dissimilarity_memmap_filename is None:
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
                                                                                           tmp_dir, dist_func,
                                                                                           process_id, process_batch_size,
                                                                                           dissimilarity_memmap_filename)))
            process_pool[process_id].start()
        for process in process_pool:
            process.join()
            process.terminate()
        
    end_time = time.time()
    total_time = (end_time-start_time)/3600.0
    total_clustering_time += total_time
    print('Done generating dissimilarity_matrix. Took {} hours'.format(total_time))
    
    # step 2: compute number of neighbors within cutoff for each instance
    print('Computing nearest neighbors for each instance...')
    start_time = time.time()
    neighbor_matrix = np.memmap(tmp_dir+'/neighbor_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                dtype='uint8', mode='w+', shape=(n_instances, n_instances))
    nn_total_vector = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(cutoff), 
                                dtype='int32', mode='w+', shape=(n_instances,))
    del nn_total_vector
    del neighbor_matrix
    process_pool = []
    examples_per_proc = n_instances//process_count
    for process_id in range(process_count): 
        start_ind = process_id*examples_per_proc
        end_ind = (process_id+1)*examples_per_proc
        if process_id == (process_count-1):
            end_ind = n_instances
            
        if start_ind >= n_instances:
            break
            
        process_pool.append(Process(target=compute_nn_total_wrapper, args=(start_ind, end_ind,
                                                                           n_instances, cutoff,
                                                                           tmp_dir, 
                                                                           process_id, process_batch_size,
                                                                           dissimilarity_memmap_filename)))
        process_pool[process_id].start()
    for process in process_pool:
        process.join()
        process.terminate()
    end_time = time.time()
    total_time = (end_time-start_time)/3600.0
    total_clustering_time += total_time
    print('Done computing nearest neighbors for each instance. Took {} hours'.format(total_time))
    
    # step 3: start clustering non-singletons
    print('Clustering non-singletons...')
    start_time = time.time()
    cluster_assigment_vector = np.memmap(output_dir+'/cluster_assigment_vector_{}.dat'.format(cutoff), 
                                         dtype='int32', mode='w+', shape=(n_instances,))
    cluster_leader_idx_vector = np.memmap(tmp_dir+'/cluster_leader_idx_vector_{}.dat'.format(cutoff), 
                                          dtype='int32', mode='w+', shape=(n_instances,))
    cluster_assigment_vector[:] = -1
    cluster_leader_idx_vector[:] = -1
    cluster_assigment_vector.flush()
    cluster_leader_idx_vector.flush()
    
    neighbor_matrix = np.memmap(tmp_dir+'/neighbor_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                dtype='uint8', mode='r+', shape=(n_instances, n_instances))
    nn_total_vector = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(cutoff), 
                                dtype='int32', mode='r+', shape=(n_instances,))
    cluster_id = 0
    row_batch_size = process_batch_size
    n_cols = n_instances
    while nn_total_vector.max() > 1:
        max_neighbor_ind = nn_total_vector.argmax()
        print(max_neighbor_ind, nn_total_vector[max_neighbor_ind])
        # gather neighbors of this instance with max number of neighbors
        neighbor_indices = []
        for batch_i in range(n_cols//row_batch_size + 1): 
            col_start = batch_i*row_batch_size
            col_end = (batch_i+1)*row_batch_size
            nm_slice = neighbor_matrix[max_neighbor_ind, col_start:col_end]
            neighbor_indices.append(col_start + np.where(nm_slice > 0)[0])
        neighbor_indices = np.hstack(neighbor_indices)
        cluster_assigment_vector[neighbor_indices] = cluster_id
        cluster_id += 1
        cluster_leader_idx_vector[neighbor_indices] = max_neighbor_ind
        
        # now remove these neighbors from other instance neighborhoods
        for idx in neighbor_indices:
            n_idx_neighbors = 0
            for batch_i in range(n_cols//row_batch_size + 1): 
                col_start = batch_i*row_batch_size
                col_end = (batch_i+1)*row_batch_size
                nm_slice = neighbor_matrix[idx, col_start:col_end]
                qualified_neighbors = np.where(nm_slice > 0)[0]
                
                if qualified_neighbors.shape[0] > 0:
                    nn_total_vector[col_start + qualified_neighbors] -= 1
                    n_idx_neighbors += qualified_neighbors.shape[0]
                    
                    # modify neighbor of idx with other indices so that it is no longer neighbor
                    neighbor_matrix[idx, col_start + qualified_neighbors] = 0
                    neighbor_matrix[col_start + qualified_neighbors, idx] = 0
            nn_total_vector[idx] -= (n_idx_neighbors-1)
        neighbor_matrix.flush()
        nn_total_vector.flush()
        cluster_leader_idx_vector.flush()
    
    del cluster_assigment_vector
    del cluster_leader_idx_vector
    del neighbor_matrix
    del nn_total_vector
    end_time = time.time()
    total_time = (end_time-start_time)/3600.0
    total_clustering_time += total_time
    print('Done clustering non-singletons. Took {} hours'.format(total_time))
    
    # step 4: start clustering singletons
    # false singletons are assigned to the same cluster of the closest instance that is within the cutoff
    print('Clustering singletons...')
    start_time = time.time()
    process_pool = []
    examples_per_proc = n_instances//process_count
    for process_id in range(process_count): 
        start_ind = process_id*examples_per_proc
        end_ind = (process_id+1)*examples_per_proc
        if process_id == (process_count-1):
            end_ind = n_instances
            
        if start_ind >= n_instances:
            break
            
        process_pool.append(Process(target=cluster_singletons_wrapper, args=(start_ind, end_ind,
                                                                             n_instances, cutoff,
                                                                             output_dir, tmp_dir, 
                                                                             cluster_id, 
                                                                             process_id, process_batch_size,
                                                                             dissimilarity_memmap_filename)))
        process_pool[process_id].start()
    for process in process_pool:
        process.join()
        process.terminate()
    end_time = time.time()
    total_time = (end_time-start_time)/3600.0
    total_clustering_time += total_time
    print('Done clustering singletons. Took {} hours'.format(total_time))
    
    print('Done clustering. Took {} hours'.format(total_clustering_time))
    
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
    parser.add_argument('--process_count', type=int, default=1, action="store", dest="process_count", required=False)
    parser.add_argument('--process_batch_size', type=int, default=2**17, action="store", dest="process_batch_size", required=False)
    parser.add_argument('--dissimilarity_memmap_filename', default=None, action="store", dest="dissimilarity_memmap_filename", required=False)
    parser.add_argument('--index_name', default='Index ID', action="store", dest="index_name", required=False)
    
    given_args = parser.parse_args()
    csv_file_or_dir = given_args.csv_file_or_dir
    output_dir = given_args.output_dir
    feature_name = given_args.feature_name
    cutoff = given_args.cutoff
    dist_function = given_args.dist_function
    process_count = given_args.process_count
    process_batch_size = given_args.process_batch_size
    dissimilarity_memmap_filename = given_args.dissimilarity_memmap_filename
    index_name = given_args.index_name

    # create tmp directory to store memmap arrays
    tmp_dir = './tmp/'
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
    
    num_files = len(glob.glob(csv_file_or_dir.format('*')))
    csv_files_list = [csv_file_or_dir.format(i) for i in range(num_files)]
    n_instances, n_features = get_features(csv_files_list, feature_name, index_name, tmp_dir, process_batch_size) 
    dist_func = feature_dist_func_dict()[dist_function]
                                     
    # cluster
    cluster_features(n_instances, n_features, dist_func, output_dir, tmp_dir, 
                     cutoff=cutoff, process_count=process_count, process_batch_size=process_batch_size,
                     dissimilarity_memmap_filename=dissimilarity_memmap_filename)
    
    # clean up tmp directory 
    #import shutil
    #shutil.rmtree(tmp_dir)