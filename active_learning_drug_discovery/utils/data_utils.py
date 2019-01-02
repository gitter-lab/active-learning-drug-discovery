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
    Computes tanimoto dissimilarity between two vectors. 
"""
def tanimoto_dissimilarity(X, Y):
    X = X.astype(bool)
    Y = Y.astype(bool)
    tan_sim = np.sum(np.bitwise_and(X, Y))/np.sum(np.bitwise_or(X, Y))
    return 1 - tan_sim
 
"""
    Returns indices of duplicated smiles from x_smiles in y_smiles. 
"""
def get_duplicate_smiles(x_smiles, y_smiles):
    x_canon_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x im x_smiles]
    y_canon_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(y)) for y im y_smiles]
    
    smiles_df = pd.DataFrame(data=[x_canon_smiles + y_canon_smiles],
                                 col=['rdkit SMILES'])
    smiles_df = smiles_df[smiles_df['rdkit SMILES'].duplicated(keep='first')]
    idx_to_drop = smiles_df.groupby(by='rdkit SMILES').apply(lambda x: list(x.index)).tolist()
    idx_to_drop = list(np.array(idx_to_drop).flatten())
    return idx_to_drop
    
"""
    Computes avg cluster dissimilarity/distance of candidate clusters towards selected clusters.
    clusters_ordered_ids is the ordering of clusters_avg_dissimilarity.
"""
def get_avg_cluster_dissimilarity(clusters_train_unlabeled, 
                                  features_train_unlabeled, 
                                  selected_cluster_ids, 
                                  candidate_cluster_ids,
                                  feature_dist_func=tanimoto_dissimilarity):
    clusters_ordered_ids = candidate_cluster_ids
    clusters_avg_dissimilarity = np.zeros(shape=(len(candidate_cluster_ids),))
    
    for selected_cid in selected_cluster_ids:
        selected_cid_instances = np.where(clusters_train_unlabeled == selected_cid)[0]
        curr_clusters_dissimilarity = np.zeros(shape=(len(candidate_cluster_ids),))
        for i, candidate_cid in enumerate(candidate_cluster_ids):
            candidate_cid_instances = np.where(clusters_train_unlabeled == candidate_cid)[0]
            avg_cluster_dist = []
            for selected_instance in selected_cid_instances:
                for candidate_instance in candidate_cid_instances:
                    avg_cluster_dist.append(feature_dist_func(features_train_unlabeled[selected_instance,:], features_train_unlabeled[candidate_instance,:]))
                    
            curr_clusters_dissimilarity[i] = np.mean(avg_cluster_dist)
            
        clusters_avg_dissimilarity += curr_clusters_dissimilarity
    
    clusters_avg_dissimilarity /= len(selected_cid)
    return clusters_ordered_ids, clusters_avg_dissimilarity