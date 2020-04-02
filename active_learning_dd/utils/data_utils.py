"""
    Contains data utils. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.FilterCatalog import *
import numpy as np
import pandas as pd

"""
    Computes tanimoto dissimilarity array between two feature matrices.
    Compares each row of X with each row of Y.
"""
def tanimoto_dissimilarity(X, Y, X_batch_size=50, Y_batch_size=50):
    n_features = X.shape[-1]
    if X.ndim == 1:
        X = X.reshape(-1, n_features)
    if Y.ndim == 1:
        Y = Y.reshape(-1, n_features)    
    tan_sim = []
    X_total_batches = X.shape[0] // X_batch_size + 1
    Y_total_batches = Y.shape[0] // Y_batch_size + 1
    for X_batch_i in range(X_total_batches):
        X_start_idx = X_batch_i*X_batch_size
        X_end_idx = min((X_batch_i+1)*X_batch_size, X.shape[0])
        X_batch = X[X_start_idx:X_end_idx,:]
        for Y_batch_i in range(Y_total_batches):
            Y_start_idx = Y_batch_i*Y_batch_size
            Y_end_idx = min((Y_batch_i+1)*Y_batch_size, Y.shape[0])
            Y_batch = Y[Y_start_idx:Y_end_idx,:]
            
            # adapted from: https://github.com/deepchem/deepchem/blob/2531eca8564c1dc68910d791b0bcd91fd586afb9/deepchem/trans/transformers.py#L752
            numerator = np.dot(X_batch, Y_batch.T).flatten() # equivalent to np.bitwise_and(X_batch, Y_batch), axis=1)
            denominator = n_features - np.dot(1-X_batch, (1-Y_batch).T).flatten() # np.sum(np.bitwise_or(X_rep, Y_rep), axis=1)
            
            tan_sim.append(numerator / denominator)
    tan_sim = np.hstack(tan_sim)
    return 1.0 - tan_sim

"""
    Computes tanimoto dissimilarity between two vectors. 
"""
def feature_dist_func_dict():
    return {"tanimoto_dissimilarity": tanimoto_dissimilarity}

    
"""
    Returns indices of duplicated smiles from x_smiles in y_smiles. 
"""
def get_duplicate_smiles_in1d(x_smiles, y_smiles, smiles_are_canonical=True):
    x_canon_smiles = x_smiles
    y_canon_smiles = y_smiles
    if not smiles_are_canonical:
        x_canon_smiles = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in x_smiles])
        y_canon_smiles = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(y)) for y in y_smiles])
    
    y_duplicates = np.in1d(y_canon_smiles, x_canon_smiles)
    idx_to_drop = list(np.arange(len(y_canon_smiles))[y_duplicates])
    
    return idx_to_drop
    
"""
    Computes avg cluster dissimilarity/distance of candidate clusters towards selected clusters.
    clusters_ordered_ids is the ordering of clusters_avg_dissimilarity.
    Assumes feature distance function returns array with distances between rows of X and Y.
"""
def get_avg_cluster_dissimilarity(clusters, 
                                  features, 
                                  selected_cluster_ids, 
                                  candidate_cluster_ids,
                                  feature_dist_func=tanimoto_dissimilarity,
                                  candidate_cluster_batch_size=2056):
    clusters_ordered_ids = candidate_cluster_ids #[:] no need to make a copy
    #clusters_avg_dissimilarity = np.zeros(shape=(len(candidate_cluster_ids),))
    
    selected_cid_instances = np.in1d(clusters, selected_cluster_ids)
    cluster_dist_means_list = []
    total_batches = candidate_cluster_ids.shape[0] // candidate_cluster_batch_size + 1
    for batch_i in range(total_batches):
        start_idx = batch_i*candidate_cluster_batch_size
        end_idx = min((batch_i+1)*candidate_cluster_batch_size, candidate_cluster_ids.shape[0])
        candidate_batch = candidate_cluster_ids[start_idx:end_idx]
        candidate_cid_instances = np.in1d(clusters, candidate_batch)
        
        candidate_cluster_rep = np.repeat(clusters[candidate_cid_instances], len(clusters[selected_cid_instances]))
        candidate_cluster_dist = feature_dist_func(features[selected_cid_instances,:], features[candidate_cid_instances,:])
        
        dist_df = pd.DataFrame(data=np.hstack([candidate_cluster_dist.reshape(-1,1),
                                               candidate_cluster_rep.reshape(-1,1)]),
                               columns=['dist', 'candidate_group'])
        cluster_dist_means_list.append(dist_df.groupby('candidate_group').mean().loc[candidate_batch].values.flatten())
    clusters_avg_dissimilarity = np.hstack(cluster_dist_means_list)
    return clusters_ordered_ids, clusters_avg_dissimilarity

"""
    ----
    curr_clusters_dissimilarity = np.zeros(shape=(len(candidate_cluster_ids),))
    for selected_cid in selected_cluster_ids:
        selected_cid_instances = np.where(clusters == selected_cid)[0]
        candidate_cid_instances = np.in1d(clusters, candidate_cluster_ids)
        
        candidate_cluster_rep = np.repeat(clusters[candidate_cid_instances], len(selected_cid_instances))
        candidate_cluster_dist = feature_dist_func(features[selected_cid_instances,:], features[candidate_cid_instances,:])
        
        dist_df = pd.DataFrame(data=np.hstack([candidate_cluster_dist.reshape(-1,1), 
                                               candidate_cluster_rep.reshape(-1,1)]),
                               columns=['dist', 'group'])
        cluster_dist_means = dist_df.groupby('group').mean().values.flatten()
        
        sorted_idx = np.argsort(candidate_cluster_ids)
        rev_sorted_idx = np.zeros(len(candidate_cluster_ids), dtype=int)
        rev_sorted_idx[sorted_idx] = np.arange(len(candidate_cluster_ids)) # adapted from: https://stackoverflow.com/a/10831155
        curr_clusters_dissimilarity[:] = cluster_dist_means[rev_sorted_idx]
        
        clusters_avg_dissimilarity += curr_clusters_dissimilarity
    
    clusters_avg_dissimilarity /= len(selected_cluster_ids)
"""    

"""
    Computes avg cluster dissimilarity/distance of candidate clusters towards selected clusters.
    Uses a disk-stored np.memmap matrix storing the instance dissimilarities.
"""
def get_avg_cluster_dissimilarity_from_file(clusters, 
                                            memmap_filename, 
                                            n_instances,
                                            selected_cluster_ids, 
                                            candidate_cluster_ids,
                                            candidate_cluster_batch_size=2056,
                                            batched_clusters_method=True):
    dissimilarity_matrix = np.memmap(memmap_filename, shape=(n_instances, n_instances), 
                                     dtype='float16', mode='r')    
    clusters_ordered_ids = candidate_cluster_ids[:]
    clusters_avg_dissimilarity = np.zeros(shape=(len(candidate_cluster_ids),))
    cluster_dist_means_list = []
    selected_cid_instances = np.in1d(clusters, selected_cluster_ids)
    
    if batched_clusters_method:
        total_batches = candidate_cluster_ids.shape[0] // candidate_cluster_batch_size + 1
        for batch_i in range(total_batches):
            start_idx = batch_i*candidate_cluster_batch_size
            end_idx = min((batch_i+1)*candidate_cluster_batch_size, candidate_cluster_ids.shape[0])
            candidate_batch = candidate_cluster_ids[start_idx:end_idx]
            candidate_cid_instances = np.in1d(clusters, candidate_batch)
            
            candidate_cluster_rep = np.repeat(clusters[candidate_cid_instances], len(clusters[selected_cid_instances]))
            dm_slice = dissimilarity_matrix[candidate_cid_instances, :][:,selected_cid_instances]
            dm_slice = dm_slice.flatten().reshape(-1,1)
            
            dist_df = pd.DataFrame(data=np.hstack([dm_slice,
                                                   candidate_cluster_rep.reshape(-1,1)]),
                                   columns=['dist', 'candidate_group'])
            cluster_dist_means_list.append(dist_df.groupby('candidate_group').mean().loc[candidate_batch].values.flatten())
    else:
        for ccid in clusters_ordered_ids:
            ccid_instances_idx = np.where(clusters == ccid)[0]
            dm_slice = dissimilarity_matrix[ccid_instances_idx, :][:,selected_cid_instances]
            cluster_dist_means_list.append(np.mean(dm_slice))
            
    clusters_avg_dissimilarity = np.hstack(cluster_dist_means_list)
    del dissimilarity_matrix
    return clusters_ordered_ids, clusters_avg_dissimilarity
    
"""
    Computes dissimilarity matrix for a given row of features.
"""
def get_dissimilarity_matrix(features, 
                             feature_dist_func=tanimoto_dissimilarity):
    row_count = features.shape[0]
    dissimilarity_matrix = np.zeros(shape=(row_count, row_count))
    for i in range(row_count):
        for j in range(row_count):
            dissimilarity_matrix[i,j] = feature_dist_func(features[i:i+1,:], features[j:j+1,:])
    return dissimilarity_matrix
    
"""
    Returns dissimilarity matrix slice from disk-stored np.memmap matrix. .
"""
def get_dissimilarity_matrix_from_file(instances_idx, 
                                       memmap_filename,
                                       n_instances):
    dissimilarity_matrix = np.memmap(memmap_filename, shape=(n_instances, n_instances), 
                                     dtype='float16', mode='r')
    dm_slice = dissimilarity_matrix[instances_idx, :][:,instances_idx]
    del dissimilarity_matrix
    return dm_slice