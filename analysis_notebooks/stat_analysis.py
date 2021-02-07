import numpy as np
import pandas as pd
import os

from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng

from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 20})

def setup_scmamp_rpy2():
    import warnings
    warnings.filterwarnings('ignore')
    import rpy2
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr, data
    from rpy2.robjects import r, pandas2ri
    pandas2ri.activate();

    rpy2.robjects.r['options'](warn=-1)
    install_packages = False
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    #utils.install_packages('rlang');
    if install_packages:
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        packnames = ('scmamp', 'BiocManager')

        names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        biocmanager = rpackages.importr('BiocManager')
        biocmanager.install('Rgraphviz')
        biocmanager.install('graph')

    scmamp = rpackages.importr('scmamp')
    
def plot_hs_boxplots(all_df, top_df_all, 
                     figsize=(30, 14), metric_col_box='total_hits', 
                     metric_col_point='total_hits_mean', hs_id_col='hs_id', 
                     alpha=0.05, top_k=15, title='', xlabel='', ylabel='', isExp1=False):
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
    
    all_df_sorted['hs_id'] = ["CBWS_{}".format(x.split('ClusterBasedWCSelector_')[1]) if 'ClusterBasedWCSelector' in x else x for x in all_df_sorted['hs_id'].tolist()]
    top_df_all['hs_id'] = ["CBWS_{}".format(x.split('ClusterBasedWCSelector_')[1]) if 'ClusterBasedWCSelector' in x else x for x in top_df_all['hs_id'].tolist()]
    top_k_dtk = ["CBWS_{}".format(x.split('ClusterBasedWCSelector_')[1]) if 'ClusterBasedWCSelector' in x else x for x in top_k_dtk]
    
    all_df_sorted['hs_id'] = ["*{}".format(x) if 'top' in y else x for x, y in zip(all_df_sorted['hs_id'].tolist(), all_df_sorted['hs_group'].tolist())]
    top_df_all['hs_id'] = ["*{}".format(x) if 'top' in y else x for x, y in zip(top_df_all['hs_id'].tolist(), top_df_all['hs_group'].tolist())]
    
    all_df_sorted['hs_id'] = ["+{}".format(x) if 'middle' in y else x for x, y in zip(all_df_sorted['hs_id'].tolist(), all_df_sorted['hs_group'].tolist())]
    top_df_all['hs_id'] = ["+{}".format(x) if 'middle' in y else x for x, y in zip(top_df_all['hs_id'].tolist(), top_df_all['hs_group'].tolist())]
    
    all_df_sorted['hs_id'] = ["-{}".format(x) if 'worst' in y else x for x, y in zip(all_df_sorted['hs_id'].tolist(), all_df_sorted['hs_group'].tolist())]
    top_df_all['hs_id'] = ["-{}".format(x) if 'worst' in y else x for x, y in zip(top_df_all['hs_id'].tolist(), top_df_all['hs_group'].tolist())]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x="hs_id", y=metric_col_box, data=all_df_sorted)
    sns.pointplot(x="hs_id", y=metric_col_point, data=top_df_all, linestyles='--', scale=0.8, 
                  color='k', errwidth=0, capsize=0)
    
    plt.xticks(rotation=90)
    
    if isExp1:
        [x.set_color("red") for x in ax.get_xticklabels() if x.get_text().replace('*','').replace('-','').replace('+','') in top_k_dtk]
    else:
        [x.set_color("red") for x in ax.get_xticklabels() if x.get_text() in top_k_dtk]
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
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

def get_stat_test_data_df(all_96_cs, all_96_bm, all_96_hs, metric='total_hits', topk=-1):
    from analysis_helper import get_top_summary
    top_96_hs = get_top_summary(all_96_hs)
    top_96_bm = get_top_summary(all_96_bm)
    top_96_cs = get_top_summary(all_96_cs)
    top_96_all = pd.concat([top_96_cs, top_96_bm, top_96_hs])
    
    sorted_hs_ids = top_96_all.sort_values('{}_mean'.format(metric), ascending=False).index.tolist()

    all_96 = pd.concat([all_96_cs, all_96_bm, all_96_hs])
    if topk > 0:
        all_96 = all_96[all_96['hs_id'].isin(sorted_hs_ids[:topk])]
        hs_ids = sorted_hs_ids[:topk]
    else:
        all_96 = all_96[all_96['hs_id'].isin(sorted_hs_ids)]
        hs_ids = sorted_hs_ids
        
    rf_ids = all_96['rf_id'].unique()

    data_df = pd.DataFrame(data=np.zeros(shape=(len(rf_ids),len(hs_ids))),
                      columns=hs_ids, index=rf_ids)
    for hs_id in hs_ids:
        for rf_id in rf_ids:
            tmp_df = all_96[(all_96['hs_id'] == hs_id) & (all_96['rf_id'] == rf_id)]

            # for (strategy, rf_id) that don't exist, we set it to mean of strategy runs that do exist
            if tmp_df.shape[0] > 0:
                metric_val = tmp_df[metric].iloc[0]
            else: 
                metric_val = np.nan

            data_df.loc[rf_id, hs_id] = metric_val


    data_df.index = [x.replace('ClusterBasedWCSelector', 'CBWS') for x in data_df.index]
    data_df.columns = [x.replace('ClusterBasedWCSelector', 'CBWS') for x in data_df.columns]
    
    return data_df

"""
    Computes contrast estimation based on medians in 4 steps as described in:
    Garcia et al. 2010 https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/sicidm/2010-Garcia-INS.pdf
    see pages 6-8
"""
def compute_custom_cem(all_96_cs, all_96_bm, all_96_hs, metric='total_hits', topk=-1):
    # get data in dataset (rows) vs strategy (columns) format
    data_df = get_stat_test_data_df(all_96_cs, all_96_bm, all_96_hs, metric='total_hits', topk=topk)

    # perform steps 1 and 2 of computing Zuv matrix
    num_algorithms = data_df.columns.shape[0]
    algorithm_names = data_df.columns.tolist()
    Zuv_matrix = pd.DataFrame(data=np.zeros(shape=(num_algorithms, num_algorithms)),
                              columns=algorithm_names,
                              index=algorithm_names)
    
    for u_idx in range(num_algorithms):
        for v_idx in range(u_idx+1, num_algorithms):
            u = algorithm_names[u_idx]
            v = algorithm_names[v_idx]
            
            tmp_df = data_df[[u, v]].copy()
            tmp_df = tmp_df.dropna(axis=0)
            u_arr = tmp_df[u].values 
            v_arr = tmp_df[v].values

            # get difference vector of strategies u and v
            perf_diff = u_arr - v_arr

            # get median differences
            median_diff = np.median(perf_diff)

            # save to Zuv matrix
            Zuv_matrix.loc[u,v] = median_diff
            Zuv_matrix.loc[v,u] = -median_diff
            
    # step 3 compute mean of median differens
    mean_medians_diff = Zuv_matrix.mean(axis=1)

    # step 4 compute difference of strategy u and v
    cem_matrix = pd.DataFrame(data=np.zeros(shape=(num_algorithms, num_algorithms)),
                              columns=algorithm_names,
                              index=algorithm_names)
    for u_idx in range(num_algorithms):
        for v_idx in range(u_idx+1, num_algorithms):
            u = algorithm_names[u_idx]
            v = algorithm_names[v_idx]
            u_val = mean_medians_diff.loc[u]
            v_val = mean_medians_diff.loc[v]

            # save to Zuv matrix
            cem_matrix.loc[u,v] = u_val - v_val
            cem_matrix.loc[v,u] = v_val - u_val
            
    return cem_matrix

def get_friedman_test(all_96_cs, all_96_bm, all_96_hs, metric='total_hits', topk=-1):
    setup_scmamp_rpy2()
    import rpy2.robjects.packages as rpackages
    scmamp = rpackages.importr('scmamp')

    data_df = get_stat_test_data_df(all_96_cs, all_96_bm, all_96_hs, metric, topk)

    res = scmamp.imanDavenportTest(data_df)
    corr_friedman_chisq = res[0][0]
    df1, df2 = res[1]
    pval = res[2][0]
    
    return corr_friedman_chisq, df1, pval

def plot_cem_heatmap(all_96_cs, all_96_bm, all_96_hs, figsize, title, 
                     metric='total_hits', topk=-1, use_custom_cem=False, fontsize=40):
    from matplotlib.collections import QuadMesh
    from matplotlib.text import Text
    
    cem_df = compute_custom_cem(all_96_cs, all_96_bm, all_96_hs, metric, topk)
    
    heatmap_df = cem_df.copy()
    heatmap_df[' '] = 0
    heatmap_df['  '] = 0
    heatmap_df['Total Wins'] = (cem_df > 0).sum(axis=1)
    heatmap_df = heatmap_df.sort_values('Total Wins', ascending=False)
    heatmap_df = heatmap_df[heatmap_df.index.tolist()+[' ', '  ', 'Total Wins']]
    
    labels = []
    for x in heatmap_df['Total Wins'].values:
        tmp = ['' for _ in range(heatmap_df.shape[0])] + ['', '', '{}'.format(x)]
        labels.append(tmp)
    labels = np.array(labels)

    fig, ax = plt.subplots(1, 1, figsize=(32, 30))
    
    cmap = plt.get_cmap("RdYlGn")
    sns.heatmap(heatmap_df, annot=labels, fmt='', square=True, cbar_kws={"shrink": 0.75}, 
                cmap=cmap, linewidths=0.1, linecolor='gray')

    # find your QuadMesh object and get array of colors
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # make colors of the last column white
    facecolors[np.arange(heatmap_df.shape[1]-1, heatmap_df.shape[0]*heatmap_df.shape[1], heatmap_df.shape[1])] = np.array([1,1,1,1])
    facecolors[np.arange(heatmap_df.shape[1]-2, heatmap_df.shape[0]*heatmap_df.shape[1], heatmap_df.shape[1])] = np.array([1,1,1,1])
    facecolors[np.arange(heatmap_df.shape[1]-3, heatmap_df.shape[0]*heatmap_df.shape[1], heatmap_df.shape[1])] = np.array([1,1,1,1])

    # set modified colors
    quadmesh.set_facecolors = facecolors

    # set color of all text to black
    for i in ax.findobj(Text):
        i.set_color('black')
    
    plt.title(title, fontsize=fontsize, y=1.035)
    
    locs, _ = plt.xticks()
    locs = [i+0.37 for i in locs]
    plt.xticks(locs, heatmap_df.columns.tolist(), rotation=70, ha='right')
    
    plt.show()
    
    cem_wins_df = heatmap_df['Total Wins']
    cem_wins_df.name = 'Total Wins {}'.format(metric)
    return cem_df, cem_wins_df

def plot_boxplots_simple(all_df, top_df_all, 
                         figsize=(30, 14), metric_col_box='total_hits', 
                         metric_col_point='total_hits_mean', hs_id_col='hs_id', 
                         title='', xlabel='', ylabel='', isExp1=False, fontsize=40):
    all_df.index = all_df[hs_id_col]
    top_df_all = top_df_all.sort_values(metric_col_point, ascending=False)
    top_df_all[hs_id_col] = top_df_all.index.tolist()

    all_df_sorted = all_df[all_df['iter_num'] == 9999].loc[top_df_all[hs_id_col].tolist(),:]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x="hs_id", y=metric_col_box, data=all_df_sorted)
    sns.pointplot(x="hs_id", y=metric_col_point, data=top_df_all, linestyles='--', scale=0.8, 
                  color='k', errwidth=0, capsize=0)
    labels = [x.replace('ClusterBasedWCSelector', 'CBWS') for x in top_df_all[hs_id_col].tolist()]
    locs, _ = plt.xticks()
    locs = [i+0.2 for i in locs]
    plt.xticks(locs, labels, rotation=90)
    
    if isExp1:
        hs_groups = top_df_all['hs_group'].tolist()
        ax_xticks = ax.get_xticklabels()
        [x.set_color("green") for x, y in zip(ax_xticks, hs_groups) if 'top' in y]
        [x.set_color("blue") for x, y in zip(ax_xticks, hs_groups) if 'middle' in y]
        [x.set_color("red") for x, y in zip(ax_xticks, hs_groups) if 'worst' in y]
        
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                               markerfacecolor='g', markersize=15),
                        Line2D([0], [0], marker='o', color='w', label='Scatter',
                               markerfacecolor='b', markersize=15),
                        Line2D([0], [0], marker='o', color='w', label='Scatter',
                               markerfacecolor='r', markersize=15)]
    
        ax.legend(custom_lines, ['Top', 'Middle', 'Bottom'],
                  title='Experiment 0 Sampled\nCBWS Color Code:', title_fontsize=35, fontsize=35)
            
    plt.xticks(rotation=70, ha='right')
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize, y=1.035)
    
    plt.show()
    