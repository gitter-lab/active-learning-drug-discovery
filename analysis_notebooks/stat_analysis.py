import numpy as np
import pandas as pd
import os

from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng

from scipy.stats import spearmanr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 20})

def plot_hs_boxplots(all_df, top_df_all, 
                     figsize=(30, 14), metric_col_box='total_hits', 
                     metric_col_point='total_hits_mean', hs_id_col='hs_id', 
                     alpha=0.05, top_k=15):
    # perform dtk on metric
    hs_ids = all_df[hs_id_col].unique()
    metric_dtk = get_dtk_df(all_df, metric_col=metric_col_box, alpha=alpha)

    metric_wins = get_dtk_win_sum(metric_dtk, hs_ids)

    summed_wins = (metric_wins).sort_values(ascending=False)
    top_k_dtk = summed_wins.index[:top_k].tolist()
    
    # plot boxplots with red xticks for top_k from dtk wins
    all_df.index = all_df[hs_id_col]
    top_df_all = top_df_all.sort_values(metric_col_point, ascending=False)
    top_df_all[hs_id_col] = top_df_all.index.tolist()

    all_df_sorted = all_df[all_df['iter_num'] == 9999].loc[list(top_df_all[hs_id_col]),:]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x="hs_id", y=metric_col_box, data=all_df_sorted)
    sns.pointplot(x="hs_id", y=metric_col_point, data=top_df_all, linestyles='--', scale=0.8, 
                  color='k', errwidth=0, capsize=0)

    plt.xticks(rotation=90)
    
    [x.set_color("red") for x in ax.get_xticklabels() if x.get_text() in top_k_dtk]
    
    plt.show()
    
    return summed_wins
    
def get_dtk_df(all_df, metric_col, hs_id_col='hs_id', alpha=0.05):
    all_df = all_df[all_df['iter_num'] == 9999]

    model_names_rep = all_df[hs_id_col].tolist()
    m_df_mat = all_df[metric_col].values
    m_df_mat = np.around(m_df_mat, decimals=4)

    dtk_lib = rpackages.importr('DTK')
    dtk_results = dtk_lib.DTK_test(robjects.FloatVector(m_df_mat), robjects.FactorVector(model_names_rep), alpha)
    dtk_results = dtk_results[1]
    row_names = dtk_results.rownames
    dtk_array = np.array(dtk_results)

    group1 = [x.split('-')[0] for x in row_names]
    group2 = [x.split('-')[1] for x in row_names]
    dtk_df = pd.DataFrame(data=[group1, group2, 
                                list(dtk_array[:,0]),list(dtk_array[:,1]),list(dtk_array[:,2]), 
                                [False for _ in range(len(group1))]]).T
    dtk_df.columns = ['group1', 'group2', 'meandiff', 'Lower CI', 'Upper CI', 'reject'] 

    for j in range(dtk_df.shape[0]):      
        if dtk_df.iloc[j,3] > 0 or dtk_df.iloc[j,4] < 0:
            dtk_df.iloc[j,5] = True
            
    return dtk_df

def get_dtk_win_sum(dtk_results, hs_ids, hs_id_col='hs_id'):
    win_dtk_matrix = pd.DataFrame(data=np.zeros(shape=(len(hs_ids), len(hs_ids))),
                                                columns=hs_ids, index=hs_ids)

    for i, row in dtk_results[dtk_results['reject']].iterrows():
        if row['meandiff'] > 0:
            win_dtk_matrix.loc[row['group1'],row['group2']] += 1
        else:
            win_dtk_matrix.loc[row['group2'],row['group1']] += 1
            
    win_dtk_matrix_sum = win_dtk_matrix.sum(axis=1)
    return win_dtk_matrix_sum


def plot_parameter_hist(config_df):
    from pandas.api.types import is_numeric_dtype

    c_df = config_df.drop('MABSelector_2').drop('uncertainty_alpha', axis=1).drop('rnd_seed.1', axis=1)
    for col in c_df.columns[2:]:
        if col in ['feature_dist_func', 'exploration_strategy', 'hyperparameter_group', 
                   'hyperparameter_id', 'rnd_seed', 'uncertainty_method', 'use_consensus_distance']:
            continue
        temp_df = c_df[col]
        if not is_numeric_dtype(temp_df):
            temp_df = temp_df.astype('float16')

        u, counts = np.unique(temp_df, return_counts=True)
        plt.figure(figsize=(8,4))
        plt.xticks(temp_df.unique()); plt.yticks(np.arange(np.max(counts)+1)); plt.grid(axis='y')
        plt.hist(temp_df); plt.xlabel(col); 
        plt.show()