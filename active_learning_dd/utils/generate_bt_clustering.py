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
        --csv_file_or_dir=../datasets/file_*.csv \
        --output_dir=../datasets/ \
        --feature_name="Morgan FP_2_1024" \
        --cutoff=0.3 \
        --dist_function=tanimoto_dissimilarity \
        --process_count=$process_count
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

def get_features(csv_files_list, feature_name, tmp_dir) :
    # first get n_instances
    instances_per_file = []
    for f in csv_files_list:
        for chunk in pd.read_csv(f, chunksize=2**17):
            instances_per_file.append(chunk.shape[0])
            
    n_features = len(chunk[feature_name].iloc[0])
    n_instances = np.sum(instances_per_file)
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='w+', shape=(n_instances, n_features))
    chunksize = 2**17
    for i, f in enumerate(csv_files_list):
        for chunk in pd.read_csv(f, chunksize=chunksize):
            for batch_i in range(instances_per_file[i]//chunksize + 1): 
                row_start = batch_i*chunksize
                row_end = min(instances_per_file[i], (batch_i+1)*chunksize)
                if i > 0:
                    row_start = np.sum(instances_per_file[:i]) + batch_i*chunksize
                    row_end = min(np.sum(instances_per_file[:(i+1)]), np.sum(instances_per_file[:i]) + (batch_i+1)*chunksize)
                X[row_start:row_end,:] = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in chunk[feature_name]]).astype(float) # this is from: https://stackoverflow.com/a/29091970
    X.flush()
    return n_instances, n_features

"""
    Function wrapper method for computing dissimilarity_matrix for a range of indices.
    Used with multiprocessing.
"""
def compute_dissimilarity_matrix_wrapper(start_ind, end_ind,
                                         n_instanes, n_features,
                                         tmp_dir,
                                         process_id):
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='r', shape=(n_instances, n_features))
    dissimilarity_matrix_slice = np.memmap(tmp_dir+'/dissimilarity_matrix_slice_{}.dat'.format(process_id), 
                                           dtype='float16', mode='w+', shape=((end_ind-start_ind), (end_ind-start_ind)))
    for c, col in enumerate(range(start_ind, end_ind)): 
        for r, row in enumerate(range(col)): 
            X_col = X[col] 
            X_row = X[row] 
            dist_col_row = dist_func(X_col, X_row) 
            dissimilarity_matrix_slice[r, c] = dist_col_row 
            dissimilarity_matrix_slice[c, r] = dist_col_row 
    dissimilarity_matrix_slice.flush()

"""
    Function wrapper method for computing total number of nearest neigbors within cutoff for a set of instances.
    Used with multiprocessing.
"""
def compute_nn_total_wrapper(start_ind, end_ind,
                             n_instances, tmp_dir, 
                             process_id):
    dissimilarity_matrix = np.memmap(tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                     dtype='float16', mode='r', shape=(n_instances, n_instances))
    nn_total_vector_slice = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(process_id), 
                                      dtype='int32', mode='w+', shape=((end_ind-start_ind),))
    row_batch_size = 2**17
    n_cols = n_instances
    for r, row in enumerate(range(start_ind, end_ind)): 
        row_total_neighbors = 0
        for batch_i in range(n_cols//row_batch_size + 1): 
            col_start = batch_i*row_batch_size
            col_end = (batch_i+1)*row_batch_size
            dm_slice = dissimilarity_matrix[row, col_start:col_end]
            row_total_neighbors += np.sum(dm_slice <= cutoff)
        nn_total_vector_slice[r] = row_total_neighbors
    nn_total_vector_slice.flush()

"""
    Function wrapper method for clustering singletons.
    Used with multiprocessing.
"""
def cluster_singletons_wrapper_old(start_ind, end_ind,
                                   nn_total_vector, dissimilarity_matrix, 
                                   cutoff, cluster_assigment_vector,
                                   cluster_id_start, cluster_leader_idx_vector,
                                   process_id):
    cluster_id = cluster_id_start
    row_batch_size = 2**17
    n_cols = dissimilarity_matrix.shape[1]
    for row in range(start_ind, end_ind): 
        if nn_total_vector[row] == 1:
            row_total_neighbors = 0
            closest_neighbor_idx = row
            closest_neighbor_dist = 1.0
            for batch_i in range(n_cols//row_batch_size + 1): 
                col_start = batch_i*row_batch_size
                col_end = (batch_i+1)*row_batch_size
                if col_start <= closest_neighbor_idx and closest_neighbor_idx < col_end:
                    dm_slice_1 = dissimilarity_matrix[row, col_start:row]
                    dm_slice_2 = dissimilarity_matrix[row, (row+1):col_end]
                    try:
                        min_idx_slice = np.argmin(dm_slice_1[dm_slice_1 <= cutoff])
                        leader_idx = cluster_leader_idx_vector[col_start + min_idx_slice]
                        cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                        if cluster_leader_dist < closest_neighbor_dist:
                            closest_neighbor_idx = leader_idx
                            closest_neighbor_dist = cluster_leader_dist
                    except:
                        pass
                    try:
                        min_idx_slice = np.argmin(dm_slice_2[dm_slice_2 <= cutoff])
                        leader_idx = cluster_leader_idx_vector[(row+1) + min_idx_slice]
                        cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                        if cluster_leader_dist < closest_neighbor_dist:
                            closest_neighbor_idx = leader_idx
                            closest_neighbor_dist = cluster_leader_dist
                    except:
                        pass
                else:
                    dm_slice = dissimilarity_matrix[row, col_start:col_end]
                    try:
                        min_idx_slice = np.argmin(dm_slice[dm_slice <= cutoff])
                        leader_idx = cluster_leader_idx_vector[col_start + min_idx_slice]
                        cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                        if cluster_leader_dist < closest_neighbor_dist:
                            closest_neighbor_idx = leader_idx
                            closest_neighbor_dist = cluster_leader_dist
                    except:
                        pass
            
            if closest_neighbor_idx == row: # true singleton case -> assign to unique cluster
                cluster_assigment_vector[row] = cluster_id
                cluster_id += 1
            else: # false singleton case -> assign to cluster of closest cluster leader
                cluster_assigment_vector[row] = cluster_assigment_vector[closest_neighbor_idx]
    cluster_assigment_vector.flush()

"""
    Function wrapper method for clustering singletons.
    Used with multiprocessing.
"""
def cluster_singletons_wrapper(start_ind, end_ind,
                               n_instances, tmp_dir, 
                               cutoff, cluster_id_start, 
                               process_id):
    dissimilarity_matrix = np.memmap(tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                     dtype='float16', mode='r', shape=(n_instances, n_instances))
    nn_total_vector = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(cutoff), 
                                dtype='int32', mode='r', shape=(n_instances,))
    cluster_assigment_vector = np.memmap(output_dir+'/cluster_assigment_vector_{}.dat'.format(cutoff), 
                                         dtype='int32', mode='r', shape=(n_instances,))
    cluster_leader_idx_vector = np.memmap(tmp_dir+'/cluster_leader_idx_vector_{}.dat'.format(cutoff), 
                                          dtype='uint8', mode='r', shape=(n_instances,))
                                          
    singleton_cluster_assigment_vector = np.memmap(output_dir+'/singleton_cluster_assigment_vector_{}.dat'.format(process_id), 
                                                   dtype='int32', mode='w+', shape=((end_ind-start_ind),2))
    cluster_id = cluster_id_start
    row_batch_size = 2**17
    n_cols = n_instances
    for r, row in enumerate(range(start_ind, end_ind)): 
        singleton_cluster_assigment_vector[r,:] = -1
        if nn_total_vector[row] == 1:
            row_total_neighbors = 0
            closest_neighbor_idx = row
            closest_neighbor_dist = 1.0
            for batch_i in range(n_cols//row_batch_size + 1): 
                col_start = batch_i*row_batch_size
                col_end = (batch_i+1)*row_batch_size
                if col_start <= closest_neighbor_idx and closest_neighbor_idx < col_end:
                    dm_slice_1 = dissimilarity_matrix[row, col_start:row]
                    dm_slice_2 = dissimilarity_matrix[row, (row+1):col_end]
                    try:
                        min_idx_slice = np.argmin(dm_slice_1[dm_slice_1 <= cutoff])
                        leader_idx = cluster_leader_idx_vector[col_start + min_idx_slice]
                        cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                        if cluster_leader_dist < closest_neighbor_dist:
                            closest_neighbor_idx = leader_idx
                            closest_neighbor_dist = cluster_leader_dist
                    except:
                        pass
                    try:
                        min_idx_slice = np.argmin(dm_slice_2[dm_slice_2 <= cutoff])
                        leader_idx = cluster_leader_idx_vector[(row+1) + min_idx_slice]
                        cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                        if cluster_leader_dist < closest_neighbor_dist:
                            closest_neighbor_idx = leader_idx
                            closest_neighbor_dist = cluster_leader_dist
                    except:
                        pass
                else:
                    dm_slice = dissimilarity_matrix[row, col_start:col_end]
                    try:
                        min_idx_slice = np.argmin(dm_slice[dm_slice <= cutoff])
                        leader_idx = cluster_leader_idx_vector[col_start + min_idx_slice]
                        cluster_leader_dist = dissimilarity_matrix[row, leader_idx]
                        if cluster_leader_dist < closest_neighbor_dist:
                            closest_neighbor_idx = leader_idx
                            closest_neighbor_dist = cluster_leader_dist
                    except:
                        pass
            
            if closest_neighbor_idx == row: # true singleton case -> assign to unique cluster
                singleton_cluster_assigment_vector[r,0] = row
                singleton_cluster_assigment_vector[r,1] = cluster_id
                cluster_id += 1
            else: # false singleton case -> assign to cluster of closest cluster leader
                singleton_cluster_assigment_vector[r,0] = row
                singleton_cluster_assigment_vector[r,1] = cluster_assigment_vector[closest_neighbor_idx]
    singleton_cluster_assigment_vector.flush()
    
def cluster_features(n_instances, n_features, dist_func, output_dir, tmp_dir, 
                     cutoff=0.2, process_count=1):
    # step 1: generate dissimilarity_matrix
    print('Generating dissimilarity_matrix...')
    start_time = time.time()
    X = np.memmap(tmp_dir+'/X.dat', dtype='float16', mode='r', shape=(n_instances, n_features))
    dissimilarity_matrix = np.memmap(tmp_dir+'/dissimilarity_matrix_{}_{}.dat'.format(n_instances, n_instances), 
                                     dtype='float16', mode='w+', shape=(94857, 94857))
    for col in range(1, n_instances): 
        for row in range(col): 
            X_col = X[col] 
            X_row = X[row] 
            dist_col_row = dist_func(X_col, X_row) 
            dissimilarity_matrix[row, col] = dist_col_row 
            dissimilarity_matrix[col, row] = dist_col_row 
    dissimilarity_matrix.flush()
    end_time = time.time()
    total_time = (end_time-start_time)/3600.0
    total_clustering_time += total_time
    print('Done generating dissimilarity_matrix. Took {} hours'.format(total_time))
    
    # step 2: compute number of neighbors within cutoff for each instance
    print('Computing nearest neighbors for each instance...')
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
            
        process_pool.append(Process(target=compute_nn_total_wrapper, args=(start_ind, end_ind,
                                                                           n_instances, tmp_dir, 
                                                                           process_id)))
        process_pool[process_id].start()
    for process in process_pool:
        process.join()
        process.terminate()
    
    nn_total_vector = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(cutoff), 
                                dtype='int32', mode='w+', shape=(n_instances,))
    for process_id, _ in enumerate(process_pool): 
        start_ind = process_id*examples_per_proc
        end_ind = (process_id+1)*examples_per_proc
        nn_total_vector_slice = np.memmap(tmp_dir+'/nn_total_vector_{}.dat'.format(process_id), 
                                          dtype='int32', mode='w+', shape=((end_ind-start_ind),))
        n_slice_instances = nn_total_vector_slice.shape[0]
        batch_size = 2**17
        for batch_i in range(n_slice_instances//batch_size + 1): 
            row_start = batch_i*batch_size
            row_end = (batch_i+1)*batch_size
            nn_total_vector[start_ind+row_start:start_ind+row_end] = nn_total_vector_slice[row_start:row_end]
    nn_total_vector.flush()    
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
                                          dtype='uint8', mode='w+', shape=(n_instances,))
    cluster_id = 0
    row_batch_size = 2**17
    n_cols = n_instances
    while nn_total_vector.max() > 1:
        max_neighbor_ind = nn_total_vector.argmax()
        # gather neighbors of this instance with max number of neighbors
        neighbor_indices = []
        for batch_i in range(n_cols//row_batch_size + 1): 
            col_start = batch_i*row_batch_size
            col_end = (batch_i+1)*row_batch_size
            dm_slice = dissimilarity_matrix[max_neighbor_ind, col_start:col_end]
            neighbor_indices.append(col_start + np.where(dm_slice <= cutoff)[0])
        neighbor_indices = np.hstack(neighbor_indices)
        cluster_assigment_vector[neighbor_indices] = cluster_id
        cluster_id += 1
        cluster_leader_idx_vector[neighbor_indices] = max_neighbor_ind
        
        # now remove these neighbors from other instance neighborhoods
        for idx in neighbor_indices:
            for batch_i in range(n_cols//row_batch_size + 1): 
                col_start = batch_i*row_batch_size
                col_end = (batch_i+1)*row_batch_size
                dm_slice = dissimilarity_matrix[idx, col_start:col_end]
                nn_total_vector[col_start + np.where(dm_slice <= cutoff)[0]] -= 1
    
    cluster_assigment_vector.flush()
    cluster_leader_idx_vector.flush()
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
                                                                             n_instances, tmp_dir, 
                                                                             cutoff, cluster_id, 
                                                                             process_id)))
        process_pool[process_id].start()
    for process in process_pool:
        process.join()
        process.terminate()
    
    for process_id, _ in enumerate(process_pool):
        start_ind = process_id*examples_per_proc
        end_ind = (process_id+1)*examples_per_proc
        singleton_cluster_assigment_vector = np.memmap(output_dir+'/singleton_cluster_assigment_vector_{}.dat'.format(process_id), 
                                                       dtype='int32', mode='r', shape=((end_ind-start_ind),2))
        n_slice_instances = singleton_cluster_assigment_vector.shape[0]
        batch_size = 2**17
        for batch_i in range(n_slice_instances//batch_size + 1): 
            row_start = batch_i*batch_size
            row_end = (batch_i+1)*batch_size
            singleton_cs_slice = singleton_cluster_assigment_vector[row_start:row_end, :]
            singleton_cs_slice = singleton_cs_slice[singleton_cs_slice[:,0] > 0,:]
            cluster_assigment_vector[singleton_cs_slice[:,0]] = singleton_cs_slice[:,1]
    cluster_assigment_vector.flush()
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
    
    given_args = parser.parse_args()
    csv_file_or_dir = given_args.csv_file_or_dir
    output_dir = given_args.output_dir
    feature_name = given_args.feature_name
    cutoff = given_args.cutoff
    dist_function = given_args.dist_function
    process_count = given_args.process_count

    # create tmp directory to store memmap arrays
    tmp_dir = './tmp/'
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
    
    csv_files_list = glob.glob(csv_file_or_dir.format('*'))
    n_instances, n_features = get_features(csv_files_list, feature_name, tmp_dir) 
    dist_func = feature_dist_func_dict()[dist_function]
                                     
    # cluster
    cluster_features(n_instances, n_features, dist_func, output_dir, tmp_dir, 
                     cutoff=0.2, process_count=process_count)
    
    # clean up tmp directory 
    import shutil
    shutil.rmtree(tmp_dir)