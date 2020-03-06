import pandas as pd
import numpy as np
import os
import glob
import pathlib


def stratify_target(data_df, output_dir, task_col_name, 
                    split_size=96, cluster_col_name='BT_0.4 ID', 
                    random_seed=20190918):
    np.random.seed(random_seed)
    
    data_df = data_df[~pd.isna(data_df[task_col_name])]
    data_df = data_df.sort_values('Index ID')
    data_df = data_df.reset_index(drop=True)
    data_df = data_df.dropna()
    
    active_indices = np.where(data_df[task_col_name] == 1)[0]
    inactive_indices = np.where(data_df[task_col_name] == 0)[0]
    
    u, c = np.unique(data_df[cluster_col_name], return_counts=True)
    n_clusters = u.shape[0]
    n_singletons = np.where(c == 1)[0].shape[0]
    n_singletons_with_hits = data_df[data_df[cluster_col_name].isin(u[np.where(c == 1)[0]])][task_col_name].sum()

    # target info
    print('Target {}'.format(task_col_name))
    print('Total molecules: {}, Total active: {}, Total inactive: {}, Total clusters: {}.'.format(data_df.shape[0], 
                                                                        active_indices.shape[0], 
                                                                        inactive_indices.shape[0],
                                                                        np.unique(data_df[cluster_col_name]).shape[0]))
    print('Clusters #: {}. Singletons #: {}. Singletons with hits #: {}'.format(n_clusters, 
                                                                                n_singletons,
                                                                                n_singletons_with_hits))

    # split into plates
    num_splits = data_df.shape[0]//split_size + 1
    actives_per_split = int(np.ceil(active_indices.shape[0]/num_splits))
    inactives_per_split = split_size - actives_per_split
    rnd_active_idx = np.random.permutation(active_indices)
    rnd_inactive_idx = np.random.permutation(inactive_indices)

    split_indices = np.random.permutation(num_splits)
    active_i, inactive_i = 0, 0
    total_mols, total_actives, total_inactives = 0, 0, 0
    for split_count, split_i in enumerate(split_indices):
        curr_indices = []
        curr_split_size = split_size
        if split_count == (num_splits-1):
            curr_split_size = data_df.shape[0] - split_size*(num_splits-1)

        # get random actives for this split
        for i in range(actives_per_split):
            if active_i < rnd_active_idx.shape[0]:
                curr_indices.append(rnd_active_idx[active_i])
                active_i += 1
                curr_split_size-=1

        # get random inactives for this split
        for i in range(curr_split_size):
            if inactive_i < rnd_inactive_idx.shape[0]:
                curr_indices.append(rnd_inactive_idx[inactive_i])
                inactive_i += 1

        # shuffle the selected cpds
        curr_indices = np.random.permutation(curr_indices)
        curr_split_df = data_df.iloc[curr_indices,:]

        # save the split to file
        split_mols = curr_split_df.shape[0]
        split_actives = curr_split_df[curr_split_df[task_col_name] == 1].shape[0]
        split_inactives = curr_split_df[curr_split_df[task_col_name] == 0].shape[0]
        total_mols += split_mols
        total_actives += split_actives
        total_inactives += split_inactives
        curr_split_df.to_csv(output_dir+'/unlabeled_{}.csv'.format(split_i), 
                             index=False)
        print('Split {}: Total molecules: {}, Total active: {}, Total inactive: {}'.format(split_i, split_mols, 
                                                                                           split_actives, 
                                                                                           split_inactives))

    print('Total molecules: {}, Total active: {}, Total inactive: {}.'.format(total_mols, total_actives, total_inactives))
    
    # assert correctness
    cdf = pd.concat([pd.read_csv(x) for x in glob.glob(output_dir+'/unlabeled_*.csv')])
    cdf = cdf.sort_values('Index ID')
    cdf = cdf.reset_index(drop=True)

    assert cdf.equals(data_df)
    
def stratify_target_alt(data_df, output_dir, task_col_name, 
                        num_samples=10, num_actives_in_split=1,
                        split_size=96, cluster_col_name='BT_0.4 ID', 
                        random_seed=20190918):
    np.random.seed(random_seed)
    
    data_df = data_df[~pd.isna(data_df[task_col_name])]
    data_df = data_df.sort_values('Index ID')
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    
    active_indices = np.where(data_df[task_col_name] == 1)[0]
    inactive_indices = np.where(data_df[task_col_name] == 0)[0]
    
    u, c = np.unique(data_df[cluster_col_name], return_counts=True)
    n_clusters = u.shape[0]
    n_singletons = np.where(c == 1)[0].shape[0]
    n_singletons_with_hits = data_df[data_df[cluster_col_name].isin(u[np.where(c == 1)[0]])][task_col_name].sum()

    # target info
    print('Target {}'.format(task_col_name))
    print('Total molecules: {}, Total active: {}, Total inactive: {}, Total clusters: {}.'.format(data_df.shape[0], 
                                                                        active_indices.shape[0], 
                                                                        inactive_indices.shape[0],
                                                                        np.unique(data_df[cluster_col_name]).shape[0]))
    print('Clusters #: {}. Singletons #: {}. Singletons with hits #: {}'.format(n_clusters, 
                                                                                n_singletons,
                                                                                n_singletons_with_hits))

    # split into plates
    rnd_active_idx = np.random.permutation(active_indices)
    rnd_inactive_idx = np.random.permutation(inactive_indices)

    active_i, inactive_i = 0, 0
    total_mols, total_actives, total_inactives = 0, 0, 0
    for split_i in np.arange(num_samples):
        curr_indices = []
        curr_split_size = split_size

        # get random actives for this split
        for i in range(num_actives_in_split):
            if active_i < rnd_active_idx.shape[0]:
                curr_indices.append(rnd_active_idx[active_i])
                active_i += 1
                curr_split_size-=1

        # get random inactives for this split
        for i in range(curr_split_size):
            if inactive_i < rnd_inactive_idx.shape[0]:
                curr_indices.append(rnd_inactive_idx[inactive_i])
                inactive_i += 1

        # shuffle the selected cpds
        curr_indices = np.random.permutation(curr_indices)
        curr_split_df = data_df.iloc[curr_indices,:]

        # save the split to file
        split_mols = curr_split_df.shape[0]
        split_actives = curr_split_df[curr_split_df[task_col_name] == 1].shape[0]
        split_inactives = curr_split_df[curr_split_df[task_col_name] == 0].shape[0]
        total_mols += split_mols
        total_actives += split_actives
        total_inactives += split_inactives
        curr_split_df.to_csv(output_dir+'/unlabeled_{}.csv'.format(split_i), 
                             index=False)
        print('Split {}: Total molecules: {}, Total active: {}, Total inactive: {}'.format(split_i, split_mols, 
                                                                                           split_actives, 
                                                                                           split_inactives))
    
    # shuffle the selected cpds
    curr_indices = np.hstack([rnd_active_idx[active_i:], rnd_inactive_idx[inactive_i:]])
    curr_indices = np.random.permutation(curr_indices)
    curr_split_df = data_df.iloc[curr_indices,:]

    # save the split to file
    split_mols = curr_split_df.shape[0]
    split_actives = curr_split_df[curr_split_df[task_col_name] == 1].shape[0]
    split_inactives = curr_split_df[curr_split_df[task_col_name] == 0].shape[0]
    total_mols += split_mols
    total_actives += split_actives
    total_inactives += split_inactives
    curr_split_df.to_csv(output_dir+'/unlabeled_{}.csv'.format(num_samples), 
                         index=False)
    print('Split {}: Total molecules: {}, Total active: {}, Total inactive: {}'.format(num_samples, split_mols, 
                                                                                       split_actives, 
                                                                                       split_inactives))
        
    print('Total molecules: {}, Total active: {}, Total inactive: {}.'.format(total_mols, total_actives, total_inactives))
    
    # assert correctness
    cdf = pd.concat([pd.read_csv(x) for x in glob.glob(output_dir+'/unlabeled_*.csv')])
    cdf = cdf.sort_values('Index ID')
    cdf = cdf.reset_index(drop=True)

    assert cdf.equals(data_df)
    