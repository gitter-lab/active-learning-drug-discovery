import numpy as np
import pandas as pd
import os
import glob
import shutil
from IPython.display import clear_output
import seaborn as sns
import matplotlib.pyplot as plt


def get_hit_metrics(job_dir, iter_max=10, task_col='pcba-aid624173', cluster_col='BT_0.4 ID'):
    def _get_hits_helper(iter_df):
        hits = iter_df[task_col].sum()
        return hits
    
    des_cols = ['iter_num', 
                'exploitation_hits', 'exploration_hits', 'total_hits',
                'total_unique_hits',
                'exploitation_batch_size', 'exploration_batch_size', 'total_batch_size']
    iter_results = []
    iter_dfs = [pd.read_csv(job_dir + "/training_data/iter_0.csv")]
    for iter_i in range(iter_max):
        iter_dir = job_dir + "/iter_{}/".format(iter_i)
        exploit_csv = iter_dir + 'exploitation.csv'
        explore_csv = iter_dir + 'exploration.csv'
        exploit_hits, exploit_unique_hits, exploit_batch_size = 0,0,0
        explore_hits, explore_unique_hits, explore_batch_size = 0,0,0
        
        curr_iter_dfs = []
        if os.path.exists(exploit_csv):
            exploit_df = pd.read_csv(exploit_csv)
            exploit_hits = _get_hits_helper(exploit_df)
            exploit_batch_size = exploit_df.shape[0]
            
            curr_iter_dfs.append(exploit_df)
        if os.path.exists(explore_csv):
            explore_df = pd.read_csv(explore_csv)
            explore_hits = _get_hits_helper(explore_df)
            explore_batch_size = explore_df.shape[0]
            
            curr_iter_dfs.append(explore_df)
        
        total_hits = exploit_hits + explore_hits
        total_batch_size = exploit_batch_size + explore_batch_size

        curr_iter_df = pd.concat(curr_iter_dfs)
        iter_hits = curr_iter_df[curr_iter_df[task_col] == 1] 
        # unique hits are those that belong to a cluster for which we have not found a hit in previous iters
        train_df = pd.concat(iter_dfs)
        train_hits = train_df[train_df[task_col] == 1]
        total_unique_hits = iter_hits[~iter_hits[cluster_col].isin(train_hits[cluster_col])]
        total_unique_hits = total_unique_hits[cluster_col].unique().shape[0]
        
        iter_results.append([iter_i, 
                             exploit_hits, explore_hits, total_hits,
                             total_unique_hits,
                             exploit_batch_size, explore_batch_size, total_batch_size])
        iter_dfs.extend(curr_iter_dfs)

    job_df = pd.DataFrame(iter_results, 
                          columns=des_cols)
    total_iters = job_df['iter_num'].max()
    
    iter_sums = [10, 20, 30, 40, 50]
    sums_list = []
    for i in iter_sums:
        job_slice = job_df[job_df['iter_num'] < i]
        sum_df = job_slice.sum().to_frame().T
        sums_list.append(sum_df)
    
    sums_df = pd.concat(sums_list)
    
    final_df = pd.concat([job_df, sums_df])
    iter_sums = [9000+i for i in iter_sums]
    final_df['iter_num'] = list(np.arange(iter_max)) + iter_sums
    final_df['max_iter'] = total_iters
    
    return final_df

def get_results(results_dir, iter_max=10, task_col='pcba-aid624173', cluster_col='BT_0.4 ID', run_count_threshold=5,
                check_failure=True):
    successful_jobs = []
    failed_jobs = []

    all_96 = []
    all_384 = []
    all_1536 = []
    for i, rdir in enumerate(results_dir):
        #clear_output()
        #print('{}/{}'.format(i, len(results_dir)))
        
        config_file = rdir+'config.csv'
        
        # get job identifiers
        rd_splits = rdir.split('\\')
        hs_group = rd_splits[1]
        hs_id = rd_splits[2]
        
        task_col = rd_splits[3]
        rf_id = rd_splits[4]
        batch_size = rd_splits[5]

        # check that the job completed succesfully:
        # - exactly iter_max*batch_size cpds were selected and that they have unique Index ID
        batch_cpds = glob.glob(rdir+'iter_*/expl*.csv')
        if len(batch_cpds) > 0:
            cpd_df = pd.concat([pd.read_csv(x) for x in batch_cpds])
            if cpd_df['Index ID'].unique().shape[0] < iter_max*int(batch_size.split('_')[-1]):
                print('Failed to reach 50 iters {}_{}_{}'.format(hs_id, rf_id, task_col))
            if cpd_df['Index ID'].unique().shape[0] != cpd_df.shape[0]:
                print('Failed to uniqueness condition {}_{}_{}'.format(hs_id, rf_id, task_col))
                cpd_df.to_csv('./failed.csv')
                assert False
                
            if check_failure:
                if cpd_df.shape[0] == iter_max*int(batch_size.split('_')[-1]):
                    successful_jobs.append('{}_{}_{}'.format(hs_id, rf_id, task_col))
                    assert cpd_df['Index ID'].unique().shape[0] == iter_max*int(batch_size.split('_')[-1])
                else:
                    failed_jobs.append('{}_{}_{}'.format(hs_id, rf_id, task_col))
                    continue
            else:
                if cpd_df['Index ID'].unique().shape[0] == cpd_df.shape[0]:
                    successful_jobs.append('{}_{}_{}'.format(hs_id, rf_id, task_col))
                else:
                    failed_jobs.append('{}_{}_{}'.format(hs_id, rf_id, task_col))
                    continue
                
        else:
            failed_jobs.append('{}_{}_{}'.format(hs_id, rf_id, task_col))
            continue
        
        hs_id = hs_id.replace('ClusterBasedWCSelector', 'CBWS')
        hs_id = hs_id.replace('InstanceBasedWCSelector', 'InstanceBWS')
        
        job_df = get_hit_metrics(rdir, len(glob.glob(rdir+'iter_*/')), task_col, cluster_col)
        job_df['rf_id'] = rf_id
        job_df['hs_id'] = hs_id
        job_df['hs_group'] = hs_group
        job_df['config_file'] = config_file
        job_df['task_col'] = task_col
        
        if int(batch_size.split('_')[-1]) == 96:
            all_96.append(job_df)
        elif int(batch_size.split('_')[-1]) == 384:
            all_384.append(job_df)
        else:
            all_1536.append(job_df)
        
    if len(all_96) > 0:
        all_96 = pd.concat(all_96)
    else:
        all_96 = None
        
    if len(all_384) > 0:
        all_384 = pd.concat(all_384)
    else:
        all_384 = None
    
    if len(all_1536) > 0:
        all_1536 = pd.concat(all_1536)
    else:
        all_1536 = None
    all_df = pd.concat([all_96, all_384, all_1536])
    
    return all_96, all_384, all_1536, all_df, successful_jobs, failed_jobs

def helper_agg(col):
    if col.name  in ['rf_id', 'task_col']:
        return '-'
    elif col.name in ['hs_id', 'hs_group']:
        return col.unique()[0]
    else:
        if '_std' in col.name:
            return col.std()
        else:
            return col.mean()

def get_all_failures(results_df, iter_max):
    rf_ids = results_df['rf_id'].unique().tolist()
    task_cols = results_df['task_col'].unique().tolist()
    hs_ids = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
              'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1']
    summary_df = results_df[results_df['iter_num']==iter_max]
    cbrandom = summary_df[summary_df['hs_id'] == 'ClusterBasedRandom']

    fail_success_counts = np.zeros(shape=(len(hs_ids),4))
    for task in task_cols:
        for rf_id in rf_ids:
            for i, hs_id in enumerate(hs_ids):
                temp_random = cbrandom[(cbrandom['rf_id'] == rf_id) & (cbrandom['task_col'] == task)]
                rhits, runiquehits = temp_random['total_hits'].iloc[0], temp_random['total_unique_hits'].iloc[0]
                
                temp_df = summary_df[(summary_df['rf_id'] == rf_id) & (summary_df['task_col'] == task) & (summary_df['hs_id'] == hs_id)]
                mhits, muniquehits = temp_df['total_hits'].iloc[0], temp_df['total_unique_hits'].iloc[0]
                
                hit_limit, unique_hit_limit = temp_df['hit_limit'].iloc[0], temp_df['unique_hit_limit'].iloc[0]
                
                if (mhits <= rhits) or (muniquehits <= runiquehits):
                    fail_success_counts[i,0] += 1
                
                if (mhits / hit_limit) >= 0.1:
                    fail_success_counts[i,1] += 1
                if (mhits / hit_limit) >= 0.25:
                    fail_success_counts[i,2] += 1
                if (mhits / hit_limit) >= 0.5:
                    fail_success_counts[i,3] += 1
                
    fail_success_counts = pd.DataFrame(data=fail_success_counts,
                              columns=['# failures', '# >= 0.1', '# >= 0.25', '# >= 0.5'])
    fail_success_counts['hs_id'] = hs_ids
    return fail_success_counts

def get_last_iter_summary(results_df, iter_max, group_cols = ['hs_id', 'rf_id'],
                          add_fail_success_counts=False):
    des_cols = ['hs_id', 'rf_id', 'max_iter', 'exploitation_hits', 'exploration_hits', 'total_hits',
                'total_unique_hits', 'total_batch_size', 'hs_group', 'task_col']
    sdf1 = results_df[results_df['iter_num']==iter_max][des_cols]
    sdf1 = sdf1.groupby(group_cols).agg(helper_agg).sort_values('total_hits', ascending=False)
    sorted_hid_list = sdf1.index.tolist()

    sdf2 = results_df[results_df['iter_num']==iter_max][des_cols]
    sdf2 = sdf2[[c for c in sdf2.columns if ('_hits' in c or 'hs_id' in c or 'rf_id' in c)]]
    sdf2.columns = [c.replace('hits', 'std') for c in sdf2.columns]
    sdf2 = sdf2.groupby(group_cols).agg(helper_agg).loc[sorted_hid_list]

    sdf = pd.concat([sdf1, sdf2], axis=1)
    
    if add_fail_success_counts:
        fail_success_counts = get_all_failures(results_df, iter_max)
        new_fs_cols = fail_success_counts.drop(['hs_id'], axis=1).columns.tolist()
        for col in new_fs_cols:
            sdf[col] = 0
        sdf.loc[fail_success_counts['hs_id'].values, new_fs_cols] = fail_success_counts[new_fs_cols].values
    return sdf

"""
    for exp 3.1
"""
def get_stat_test_dict_exp3(results_df, iter_max, metric='total_hits'):
    des_cols = ['hs_id', 'rf_id', 'max_iter', 'exploitation_hits', 'exploration_hits', 'total_hits',
                'total_unique_hits', 'total_batch_size', 'hs_group', 'task_col']
    results_df = results_df[results_df['iter_num']==iter_max][des_cols]
    tasks = results_df['task_col'].unique()
    rf_ids = results_df['rf_id'].unique()
    hs_ids = results_df['hs_id'].unique()
    
    task_data_df_dict = {}
    for task_col in tasks:
        data_df = pd.DataFrame(data=np.zeros((len(rf_ids),len(hs_ids))),
                               columns=hs_ids, index=rf_ids)
        
        task_df = results_df[results_df['task_col'] == task_col]
        for hs_id in hs_ids:
            for rf_id in rf_ids:
                tmp_df = task_df[(task_df['hs_id'] == hs_id) & (task_df['rf_id'] == rf_id)]
                # for (strategy, rf_id) that don't exist, we set it to mean of strategy runs that do exist
                metric_val = tmp_df[metric].iloc[0]
                data_df.loc[rf_id, hs_id] = metric_val

    
        task_data_df_dict[task_col] = data_df
        
    return task_data_df_dict

"""
    Computes contrast estimation based on medians in 4 steps as described in:
    Garcia et al. 2010 https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/sicidm/2010-Garcia-INS.pdf
    see pages 6-8
    exp 3.1
"""
def compute_custom_cem_exp3(results_df, iter_max, metric='total_hits'):
    # get data in dataset (rows) vs strategy (columns) format    
    task_data_df_dict = get_stat_test_dict_exp3(results_df, iter_max, metric)
    
    def custom_cem_helper(data_df):
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
        
    cem_task_dict = {}
    for task_col in task_data_df_dict:
        cem_data_df = task_data_df_dict[task_col]
        cem_res = custom_cem_helper(cem_data_df)
        cem_df = pd.DataFrame(cem_res, columns=cem_data_df.columns, index=cem_data_df.columns)
        cem_task_dict[task_col] = cem_df
        
    return task_data_df_dict, cem_task_dict
    
    
"""
    cem for exp 3.1
"""
def compute_scmamp_cem_exp3(results_df, iter_max, metric='total_hits'):
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from stat_analysis import setup_scmamp_rpy2
    setup_scmamp_rpy2()

    scmamp = rpackages.importr('scmamp')
    task_data_df_dict = get_stat_test_dict_exp3(results_df, iter_max, metric)
    
    cem_task_dict = {}
    for task_col in task_data_df_dict:
        data_df = task_data_df_dict[task_col]
        cem_res = scmamp.contrastEstimationMatrix(data_df)
        cem_df = pd.DataFrame(cem_res, columns=data_df.columns, index=data_df.columns)
        cem_task_dict[task_col] = cem_df
    return task_data_df_dict, cem_task_dict


"""
    Failure of a (strategy, task, rf_id) combo is defined by having total_hits or total_unique_hits 
    not exceed that of ClusterBasedRandom or InstanceBasedRandom.
"""
def get_task_failures_dict(results_df, iter_max):
    rf_ids = results_df['rf_id'].unique().tolist()
    task_cols = results_df['task_col'].unique().tolist()
    hs_ids = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
              'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1']
    summary_df = results_df[results_df['iter_num']==iter_max]
    cbrandom = summary_df[summary_df['hs_id'] == 'ClusterBasedRandom']
    ibrandom = summary_df[summary_df['hs_id'] == 'InstanceBasedRandom']
    
    fail_success_counts_dict = {}
    for task in task_cols:
        fail_success_counts = np.zeros(shape=(len(hs_ids),4))
        for rf_id in rf_ids:
            for i, hs_id in enumerate(hs_ids):
                temp_cbrandom = cbrandom[(cbrandom['rf_id'] == rf_id) & (cbrandom['task_col'] == task)]
                temp_ibrandom = ibrandom[(ibrandom['rf_id'] == rf_id) & (ibrandom['task_col'] == task)]
                rcbhits, rcbuniquehits = temp_cbrandom['total_hits'].iloc[0], temp_cbrandom['total_unique_hits'].iloc[0]
                ribhits, ribuniquehits = temp_ibrandom['total_hits'].iloc[0], temp_ibrandom['total_unique_hits'].iloc[0]
                
                temp_df = summary_df[(summary_df['rf_id'] == rf_id) & (summary_df['task_col'] == task) & (summary_df['hs_id'] == hs_id)]
                mhits, muniquehits = temp_df['total_hits'].iloc[0], temp_df['total_unique_hits'].iloc[0]
                
                hit_limit, unique_hit_limit = temp_df['hit_limit'].iloc[0], temp_df['unique_hit_limit'].iloc[0]
                
                if (mhits <= rcbhits) or (muniquehits <= rcbuniquehits) or (mhits <= ribhits) or (muniquehits <= ribuniquehits):
                    fail_success_counts[i,0] += 1
                
                if (mhits / hit_limit) >= 0.1:
                    fail_success_counts[i,1] += 1
                if (mhits / hit_limit) >= 0.25:
                    fail_success_counts[i,2] += 1
                if (mhits / hit_limit) >= 0.5:
                    fail_success_counts[i,3] += 1
                
        fail_success_counts = pd.DataFrame(data=fail_success_counts,
                                  columns=['# failures', '# >= 0.1', '# >= 0.25', '# >= 0.5'])
        fail_success_counts['hs_id'] = hs_ids
        fail_success_counts_dict[task] = fail_success_counts
    return fail_success_counts_dict
"""
    for exp 3.1
"""
def plot_cem_heatmap_exp3(cem_df, title, figsize=(16, 16), fail_success_counts=None):
    from matplotlib.collections import QuadMesh
    from matplotlib.text import Text

    add_fail_success_counts = False
    if fail_success_counts is not None:
        add_fail_success_counts = True

    heatmap_df = cem_df.copy()
    heatmap_df[' '] = 0
    heatmap_df['Total Wins'] = (cem_df > 0).sum(axis=1)
    heatmap_df = heatmap_df.sort_values('Total Wins', ascending=False)
    ordered_wins_hs_ids = heatmap_df['Total Wins'].index.tolist()
    heatmap_df = heatmap_df[ordered_wins_hs_ids + [' ', 'Total Wins']]
    facecolor_limit = 3
    shrink_factor = 0.6

    if add_fail_success_counts:
        heatmap_df['# Failures (out of 10)'] = np.nan
        heatmap_df['# >= 10%'] = np.nan
        heatmap_df['# >= 25%'] = np.nan
        heatmap_df['# >= 50%'] = np.nan
        facecolor_limit=7
        shrink_factor = 0.5
        for hs_id in fail_success_counts['hs_id'].unique():
            tmp_df = fail_success_counts[fail_success_counts['hs_id'] == hs_id]
            failures_cnt = tmp_df['# failures'].iloc[0]
            a, b, c = tmp_df['# >= 0.1'].iloc[0], tmp_df['# >= 0.25'].iloc[0], tmp_df['# >= 0.5'].iloc[0]
            heatmap_df.loc[hs_id, '# Failures (out of 10)'] = failures_cnt
            heatmap_df.loc[hs_id, '# >= 10%'] = a
            heatmap_df.loc[hs_id, '# >= 25%'] = b
            heatmap_df.loc[hs_id, '# >= 50%'] = c

    labels = []
    for i, row in heatmap_df.iterrows():
        x = row['Total Wins']
        addendum_labels = ['', '{}'.format(x)]
        if add_fail_success_counts:
            f, a, b, c = row['# Failures (out of 10)'], row['# >= 10%'], row['# >= 25%'], row['# >= 50%']
            addendum_labels += ['{}'.format(f), '{}'.format(a), '{}'.format(b), '{}'.format(c)]
        tmp = ['' for _ in range(heatmap_df.shape[0])] + addendum_labels
        labels.append(tmp)
    labels = np.array(labels)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.heatmap(heatmap_df, annot=labels, linewidths=1, linecolor='grey', 
                fmt='', square=True, cbar_kws={"shrink": shrink_factor})

    # find your QuadMesh object and get array of colors
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # make colors of the last column white

    # set modified colors
    quadmesh.set_facecolors = facecolors

    for i in range(1, facecolor_limit):
        facecolors[np.arange(heatmap_df.shape[1]-i, heatmap_df.shape[0]*heatmap_df.shape[1], 
                             heatmap_df.shape[1])] = np.array([1,1,1,1])

    # set color of all text to black
    for i in ax.findobj(Text):
        i.set_color('black')

    plt.title(title)
    plt.show()

    cem_wins_df = heatmap_df['Total Wins']
    return cem_wins_df

def plot_cem_heatmap_all_tasks_exp3(cem_task_dict, task_info, fail_success_counts_dict, 
                                    title, figsize=(16, 16), add_fail_success_counts=True,
                                    tasks_per_row=10, shrink_factor=0.1, fontsize=35, metric='Total Hits', 
                                    save_fmt='./exp3/cem/', title_y=0.55, hspace=0.2, wspace=0.2):
    
    from matplotlib.collections import QuadMesh
    from matplotlib.text import Text
    hs_ids_order = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
                    'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1',
                    'ClusterBasedRandom', 'InstanceBasedRandom']
    
    task_info = task_info.sort_values('active_ratio')
    tasks = task_info['task_col'].tolist()
    task_labels = ['{}\n{} cpds\n{} hits\n{}% hits'.format(row['task_col'].replace('pcba-', ''), 
                                    row['cpd_count'], row['hit_limit'], 
                                    row['active_ratio']) for i, row in task_info.iterrows()]
    cem_wins_dict = {}
    total_iters = int(np.ceil(len(tasks)/tasks_per_row))
    latex_lines = []
    for task_batch in range(total_iters):
        tasks_subset = tasks[task_batch*tasks_per_row:(task_batch+1)*tasks_per_row]
        curr_tasks_per_row = len(tasks_subset)
        
        if task_batch != (total_iters-1):
            fig, axes = plt.subplots(2, curr_tasks_per_row//2, figsize=figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(50, 20))
            axes = axes.flatten()[:2]
        
        for axes_i, task_col in enumerate(tasks_subset):
            task_hit_limit = task_info[task_info['task_col'] == task_col]['hit_limit'].iloc[0]
            task_title='Task: {}. Hit limit: {}.'.format(task_col, task_hit_limit)

            cem_df = cem_task_dict[task_col]
            if add_fail_success_counts:
                fail_success_counts = fail_success_counts_dict[task_col]

            heatmap_df = cem_df.copy()
            heatmap_df[' '] = 0
            heatmap_df['Total Wins'] = (cem_df > 0).sum(axis=1)
            heatmap_df = heatmap_df.loc[hs_ids_order]
            heatmap_df = heatmap_df[hs_ids_order + [' ', 'Total Wins']]
            
            #heatmap_df = heatmap_df.sort_values('Total Wins', ascending=False)
            #ordered_wins_hs_ids = heatmap_df['Total Wins'].index.tolist()
            #heatmap_df = heatmap_df[ordered_wins_hs_ids + [' ', 'Total Wins']]
    
            facecolor_limit = 3

            if add_fail_success_counts:
                heatmap_df['# Failures (out of 10)'] = np.nan
                heatmap_df['# >= 10%'] = np.nan
                heatmap_df['# >= 25%'] = np.nan
                heatmap_df['# >= 50%'] = np.nan
                facecolor_limit=7
                for hs_id in fail_success_counts['hs_id'].unique():
                    tmp_df = fail_success_counts[fail_success_counts['hs_id'] == hs_id]
                    failures_cnt = tmp_df['# failures'].iloc[0]
                    a, b, c = tmp_df['# >= 0.1'].iloc[0], tmp_df['# >= 0.25'].iloc[0], tmp_df['# >= 0.5'].iloc[0]                    
                    heatmap_df.loc[hs_id, '# Failures (out of 10)'] = failures_cnt
                    heatmap_df.loc[hs_id, '# >= 10%'] = a
                    heatmap_df.loc[hs_id, '# >= 25%'] = b
                    heatmap_df.loc[hs_id, '# >= 50%'] = c

            labels = []
            for i, row in heatmap_df.iterrows():
                x = int(row['Total Wins'])
                addendum_labels = ['', '{}'.format(x)]
                if add_fail_success_counts:
                    f, a, b, c = row['# Failures (out of 10)'], row['# >= 10%'], row['# >= 25%'], row['# >= 50%']
                    if not np.isnan(f):
                        f = int(f)
                    if not np.isnan(a):
                        a = int(a)
                    if not np.isnan(b):
                        b = int(b)
                    if not np.isnan(c):
                        c = int(c)
                        
                    addendum_labels += ['{}'.format(f), '{}'.format(a), '{}'.format(b), '{}'.format(c)]
                tmp = ['' for _ in range(heatmap_df.shape[0])] + addendum_labels
                labels.append(tmp)
            labels = np.array(labels)
            
            cmap = plt.get_cmap("RdYlGn")
            sns.heatmap(heatmap_df, annot=labels, linewidths=1, linecolor='grey', cmap=cmap,
                        fmt='', square=True, cbar_kws={"shrink": shrink_factor}, ax=axes[axes_i])

            # find your QuadMesh object and get array of colors
            quadmesh = axes[axes_i].findobj(QuadMesh)[0]
            facecolors = quadmesh.get_facecolors()

            # make colors of the last column white

            # set modified colors
            quadmesh.set_facecolors = facecolors

            for i in range(1, facecolor_limit):
                facecolors[np.arange(heatmap_df.shape[1]-i, heatmap_df.shape[0]*heatmap_df.shape[1], 
                                     heatmap_df.shape[1])] = np.array([1,1,1,1])

           
            locs = axes[axes_i].get_xticks()
            locs = [i+0.35 for i in locs]
            axes[axes_i].set_xticks(locs)
        
           # set color of all text to black
            for i in axes[axes_i].findobj(Text):
                i.set_color('black')

            axes[axes_i].set_title(task_title, y=1.06, fontsize=fontsize)
            if axes_i%2 > 0:
                axes[axes_i].set_yticks([])
            
            if (axes_i//2 > 0) or (task_batch == (total_iters-1)):
                axes[axes_i].set_xticklabels(axes[axes_i].get_xticklabels(), rotation=70, ha='right')
            else:
                if task_batch != (total_iters-1):
                    axes[axes_i].set_xticks([])
                
            cem_wins_df = heatmap_df['Total Wins']
            cem_wins_dict[task_col] = cem_wins_df

        fig.tight_layout()
        plt.suptitle(title, fontsize=fontsize, y=title_y)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        
        if save_fmt is not None:
            plt.savefig(save_fmt+'{}_{}.png'.format(metric.replace(' ', '_'), task_batch+1), bbox_inches='tight');
        
        plt.show()
        
        latex_lines.append('\\vspace*{\\fill}')
        latex_lines.append('\\begin{figure}[H]\\ContinuedFloat')
        latex_lines.append('\\centering')
        latex_lines.append('\\includegraphics[width=\\textwidth]{project_al/experiments/exp3/cem/'+'{}_{}'.format(metric.replace(' ', '_'), task_batch+1)+'.png}')
        cont_line = '\\emph{('+ '{} of {} cont.'.format(task_batch+1, total_iters) +')}}'
        latex_lines.append('\\caption[]{Experiment 3.1 per-task contrast estimation based on medians (CEM) heatmaps for \\textbf{'+metric+'} after 50 iterations along with extra columns denoting counts for various conditions. '+cont_line)
        latex_lines.append("\\end{figure}")
        latex_lines.append("\\vspace*{\\fill}")
    
    with open(save_fmt+"/latex_{}.txt".format(metric), 'w') as f:
        for line in latex_lines:
            f.write("{}\n".format(line))
            
    return cem_wins_dict

def plot_boxplots_simple_exp3(results_df, iter_max, task_info, 
                              figsize=(16, 12), metric='total_hits', 
                              title='', xlabel='', ylabel='', save_fmt=None, 
                              fontsize=35, labelpad=20, tasks_per_plot=10, legendfontsize=25):
    hue_order = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
                 'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1', 
                 'ClusterBasedRandom', 'InstanceBasedRandom']
    results_df = results_df[results_df['iter_num']==iter_max]
    task_info = task_info.sort_values('active_ratio')
    tasks = task_info['task_col'].tolist()
    task_labels = ['{}\n{} cpds\n{} hits\n{}% hits'.format(row['task_col'].replace('pcba-', ''), 
                                    row['cpd_count'], row['hit_limit'], 
                                    row['active_ratio']) for i, row in task_info.iterrows()]
    
    latex_lines = []
    total_iters = int(np.ceil(len(tasks)/tasks_per_plot))
    for task_batch in range(total_iters):
        tasks_subset = tasks[task_batch*tasks_per_plot:(task_batch+1)*tasks_per_plot]
        xtick_labels = task_labels[task_batch*tasks_per_plot:(task_batch+1)*tasks_per_plot]

        trimmed_results_df = results_df[results_df['task_col'].isin(tasks_subset)]
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x="task_col", y=metric, hue="hs_id", data=trimmed_results_df,
                    order=tasks_subset, hue_order=hue_order)

        locs, _ = plt.xticks()
        locs = [i-0.4 for i in locs]
        plt.xticks(locs, xtick_labels, ha='left')
        plt.xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
        plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
        plt.title(title + ' (plot {} of {})'.format(task_batch+1, total_iters), fontsize=fontsize, y=1.05)
        
        [plt.axvline(x+0.5, color='r', linestyle='--') for x in range(tasks_per_plot-1)] # from:https://stackoverflow.com/a/60375919
        
        ax.legend(title='Hyperparameter ID:', title_fontsize=legendfontsize, fontsize=legendfontsize)
                  
        if save_fmt is not None:
            plt.savefig(save_fmt+'boxplots_{}_{}.png'.format(metric, task_batch+1), bbox_inches='tight');
        
        plt.show()
        
        latex_lines.append('\\vspace*{\\fill}')
        latex_lines.append('\\begin{figure}[H]\\ContinuedFloat')
        latex_lines.append('\\centering')
        latex_lines.append('\\includegraphics[width=\\textwidth]{project_al/experiments/exp3/boxplots/boxplots_'+metric+'_'+str(task_batch+1)+'.png}')
        latex_lines.append('\\caption[]{Experiment 3.1 per-task \\textbf{Total Hits} boxplots after 50 iterations (102 tasks). ')
        latex_lines.append("The x-tick labels for each task include number of compounds, number of hits, and hit \\%. \\emph{(cont.)} }")
        latex_lines.append("\\end{figure}")
        latex_lines.append("\\vspace*{\\fill}")
        latex_lines.append("\\newpage")
        
    with open(save_fmt+"/latex_{}.txt".format(metric), 'w') as f:
        for line in latex_lines:
            f.write("{}\n".format(line))
        
def get_win_summary_df(task_info, cem_all_iters_metric_dict):
    tasks = task_info.sort_values('active_ratio')['task_col'].unique()
    hs_ids_order = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
                    'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1',
                    'ClusterBasedRandom', 'InstanceBasedRandom']
    iter_max_dict = {9010: 10, 9020: 20, 9030: 30, 9040: 40, 9050: 50}
    metric_dict = {'total_hits': 'Total Hits', 'total_unique_hits': 'Total Unique Hits'}

    data_1 = []
    for metric in metric_dict:
        for iter_max in iter_max_dict:
            cem_task_dict, fail_success_counts_dict = cem_all_iters_metric_dict['{}_{}'.format(metric, iter_max)]
            for task_col in tasks:
                cem_df = cem_task_dict[task_col]
                fail_success_df = fail_success_counts_dict[task_col]
                cem_wins_df = (cem_df > 0).sum(axis=1)

                top_strategy = cem_wins_df[cem_wins_df == cem_wins_df.max()]    
                top_strategy = "|".join(top_strategy.index.tolist())
                data_1.append([metric_dict[metric], iter_max_dict[iter_max], task_col, top_strategy])

    metric_task_top_strats_df = pd.DataFrame(data=data_1, columns=['Metric', '# Iterations', 'Task', 'Best Strategy (Ties)'])

    hs_ids_order = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
                    'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1',
                    'ClusterBasedRandom', 'InstanceBasedRandom']
    tmp_df = metric_task_top_strats_df

    win_summary_df_list = []
    for metric in ['Total Hits', 'Total Unique Hits']:
        data = np.zeros(shape=(len(hs_ids_order),5))
        columns = [[],[]]
        best_per_iter = []
        for i, iter_max in enumerate(iter_max_dict):
            df = tmp_df[(tmp_df['Metric'] == metric) & (tmp_df['# Iterations'] == iter_max_dict[iter_max])]
            assert df.shape[0] == 102
            hs_id_counts = []
            for hs_id in hs_ids_order:
                hs_id_counts.append(df[df['Best Strategy (Ties)'].str.contains(hs_id)].shape[0])
            hs_id_counts = np.array(hs_id_counts)
            columns[0].append(metric)
            columns[1].append(iter_max_dict[iter_max])
            data[:,i] = hs_id_counts
            best_per_iter.append("|".join([hs_ids_order[i] for i in np.where(hs_id_counts == np.max(hs_id_counts))[0]]))
        data = list(data) + [best_per_iter]
        data_df = pd.DataFrame(data=data, columns=columns, index=hs_ids_order+['Best'])
        win_summary_df_list.append(data_df)
    
    hs_ids_order = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
                    'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1']
    fs_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                     columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    s10_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                          columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    s25_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                          columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    s50_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                          columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    for iter_max in iter_max_dict:
        _, fail_success_counts_dict = cem_all_iters_metric_dict['total_hits_{}'.format(iter_max)]

        fs_counts = np.zeros(shape=(4, len(tasks), len(hs_ids_order)))
        for i, task_col in enumerate(tasks):
            task_df = fail_success_counts_dict[task_col]
            task_df.index = task_df['hs_id'].tolist()
            task_df = task_df.drop('hs_id', axis=1)
            for j, hs_id in enumerate(hs_ids_order):
                f, a, b, c = task_df.loc[hs_id]

                fs_counts[0,i,j] = f
                fs_counts[1,i,j] = a
                fs_counts[2,i,j] = b
                fs_counts[3,i,j] = c

        fs_df[iter_max_dict[iter_max]] = fs_counts[0,:].sum(axis=0)
        s10_df[iter_max_dict[iter_max]] = fs_counts[1,:].sum(axis=0)
        s25_df[iter_max_dict[iter_max]] = fs_counts[2,:].sum(axis=0)
        s50_df[iter_max_dict[iter_max]] = fs_counts[3,:].sum(axis=0)

    fs_df = fs_df.astype(int);s10_df = s10_df.astype(int);s25_df = s25_df.astype(int);s50_df = s50_df.astype(int);
    fs_df.columns = [['# Failures for all 1020 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    s10_df.columns = [['# $\ge$ 10\% hits for all 1020 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    s25_df.columns = [['# $\ge$ 25\% hits for all 1020 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    s50_df.columns = [['# $\ge$ 50\% hits  for all 1020 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    
    [win_summary_df_list.append(df) for df in [fs_df, s10_df, s25_df, s50_df]]
    
    for iter_max in [9050]:
        _, fail_success_counts_dict = cem_all_iters_metric_dict['total_hits_{}'.format(iter_max)]

        fs_counts = np.zeros(shape=(2, len(tasks), len(hs_ids_order)))
        for i, task_col in enumerate(tasks):
            task_df = fail_success_counts_dict[task_col]
            task_df.index = task_df['hs_id'].tolist()
            task_df = task_df.drop('hs_id', axis=1)
            for j, hs_id in enumerate(hs_ids_order):
                f, a, b, c = task_df.loc[hs_id]

                fs_counts[0,i,j] = f
                fs_counts[1,i,j] = c

        fs_task_df = pd.DataFrame(data=fs_counts[0,:], columns=hs_ids_order, index=tasks)
        fs_task_df['Task Total'] = fs_task_df.sum(axis=1)
        fs_task_df = fs_task_df[fs_task_df['Task Total'] > 0].astype(int)
        fs_task_df['Task Hit %'] = [task_info[task_info['task_col'] == x]['active_ratio'].iloc[0] for x in fs_task_df.index.tolist()]
        fs_task_df = fs_task_df.sort_values('Task Hit %')
        fs_task_df = fs_task_df[['Task Hit %'] + hs_ids_order + ['Task Total']]
        fs_task_df = pd.concat([fs_task_df, fs_task_df.sum(axis=0).to_frame().T])
        fs_task_df.name = '# iterations: 50'
        fs_task_df.index = fs_task_df.index.tolist()[:-1] + ['Strategy Total']
        
        s50_task_df = pd.DataFrame(data=fs_counts[1,:], columns=hs_ids_order, index=tasks)
        s50_task_df['Task Total'] = s50_task_df.sum(axis=1)
        s50_task_df = s50_task_df[s50_task_df['Task Total'] > 0].astype(int)
        s50_task_df['Task Hit %'] = [task_info[task_info['task_col'] == x]['active_ratio'].iloc[0] for x in s50_task_df.index.tolist()]
        s50_task_df = s50_task_df.sort_values('Task Hit %')
        s50_task_df = s50_task_df[['Task Hit %'] + hs_ids_order + ['Task Total']]
        s50_task_df = pd.concat([s50_task_df, s50_task_df.sum(axis=0).to_frame().T])
        s50_task_df.name = '# iterations: 50'
        s50_task_df.index = s50_task_df.index.tolist()[:-1] + ['Strategy Total']
        
    return win_summary_df_list, fs_task_df, s50_task_df

def get_exp3_2_failures(results_df, task_info):
    tasks = task_info.sort_values('active_ratio')['task_col'].unique()
    hs_ids_order = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
                    'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1']
    iter_max_dict = {9010: 10, 9020: 20, 9030: 30, 9040: 40, 9050: 50}
    fs_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                         columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    s10_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                          columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    s25_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                          columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    s50_df = pd.DataFrame(data=np.zeros(shape=(len(hs_ids_order),5)), 
                          columns=[10, 20, 30, 40, 50], index=hs_ids_order)
    for iter_max in iter_max_dict:
        fail_success_counts_dict = get_task_failures_dict(results_df, iter_max)

        fs_counts = np.zeros(shape=(4, len(tasks), len(hs_ids_order)))
        for i, task_col in enumerate(tasks):
            task_df = fail_success_counts_dict[task_col]
            task_df.index = task_df['hs_id'].tolist()
            task_df = task_df.drop('hs_id', axis=1)
            for j, hs_id in enumerate(hs_ids_order):
                f, a, b, c = task_df.loc[hs_id]

                fs_counts[0,i,j] = f
                fs_counts[1,i,j] = a
                fs_counts[2,i,j] = b
                fs_counts[3,i,j] = c

        fs_df[iter_max_dict[iter_max]] = fs_counts[0,:].sum(axis=0)
        s10_df[iter_max_dict[iter_max]] = fs_counts[1,:].sum(axis=0)
        s25_df[iter_max_dict[iter_max]] = fs_counts[2,:].sum(axis=0)
        s50_df[iter_max_dict[iter_max]] = fs_counts[3,:].sum(axis=0)

    fs_df = fs_df.astype(int);s10_df = s10_df.astype(int);s25_df = s25_df.astype(int);s50_df = s50_df.astype(int);
    fs_df.columns = [['# Failures for all 102 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    s10_df.columns = [['# $\ge$ 10\% hits for all 102 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    s25_df.columns = [['# $\ge$ 25\% hits for all 102 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    s50_df.columns = [['# $\ge$ 50\% hits  for all 102 task-plate runs' for _ in range(5)], [10, 20, 30, 40, 50]]
    
    fs_df_list = [fs_df, s10_df, s25_df, s50_df]
    
    for iter_max in [9050]:
        fail_success_counts_dict = get_task_failures_dict(results_df, iter_max)

        fs_counts = np.zeros(shape=(2, len(tasks), len(hs_ids_order)))
        for i, task_col in enumerate(tasks):
            task_df = fail_success_counts_dict[task_col]
            task_df.index = task_df['hs_id'].tolist()
            task_df = task_df.drop('hs_id', axis=1)
            for j, hs_id in enumerate(hs_ids_order):
                f, a, b, c = task_df.loc[hs_id]

                fs_counts[0,i,j] = f
                fs_counts[1,i,j] = c

        fs_task_df = pd.DataFrame(data=fs_counts[0,:], columns=hs_ids_order, index=tasks)
        fs_task_df['Task Total'] = fs_task_df.sum(axis=1)
        fs_task_df = fs_task_df[fs_task_df['Task Total'] > 0].astype(int)
        fs_task_df['Task Hit %'] = [task_info[task_info['task_col'] == x]['active_ratio'].iloc[0] for x in fs_task_df.index.tolist()]
        fs_task_df = fs_task_df.sort_values('Task Hit %')
        fs_task_df = fs_task_df[['Task Hit %'] + hs_ids_order + ['Task Total']]
        fs_task_df = pd.concat([fs_task_df, fs_task_df.sum(axis=0).to_frame().T])
        fs_task_df.name = '# iterations: 50'
        fs_task_df.index = fs_task_df.index.tolist()[:-1] + ['Strategy Total']
        
        s50_task_df = pd.DataFrame(data=fs_counts[1,:], columns=hs_ids_order, index=tasks)
        s50_task_df['Task Total'] = s50_task_df.sum(axis=1)
        s50_task_df = s50_task_df[s50_task_df['Task Total'] > 0].astype(int)
        s50_task_df['Task Hit %'] = [task_info[task_info['task_col'] == x]['active_ratio'].iloc[0] for x in s50_task_df.index.tolist()]
        s50_task_df = s50_task_df.sort_values('Task Hit %')
        s50_task_df = s50_task_df[['Task Hit %'] + hs_ids_order + ['Task Total']]
        s50_task_df = pd.concat([s50_task_df, s50_task_df.sum(axis=0).to_frame().T])
        s50_task_df.name = '# iterations: 50'
        s50_task_df.index = s50_task_df.index.tolist()[:-1] + ['Strategy Total']
        
    return fs_df_list, fs_task_df, s50_task_df

"""
    for exp 3.2
"""
def get_stat_test_dict_exp3_2(results_df, iter_max, metric='total_hits'):
    des_cols = ['hs_id', 'rf_id', 'max_iter', 'exploitation_hits', 'exploration_hits', 'total_hits',
                'total_unique_hits', 'total_batch_size', 'hs_group', 'task_col']
    results_df = results_df[results_df['iter_num']==iter_max][des_cols]
    tasks = results_df['task_col'].unique()
    rf_ids = results_df['rf_id'].unique()
    hs_ids = results_df['hs_id'].unique()
    
    data_df = pd.DataFrame(data=np.zeros((len(tasks),len(hs_ids))),
                           columns=hs_ids, index=tasks)
    for hs_id in hs_ids:
        for task_col in tasks:
            tmp_df = results_df[(results_df['hs_id'] == hs_id) & (results_df['task_col'] == task_col)]
            assert tmp_df.shape[0] == 1
            metric_val = tmp_df[metric].iloc[0]
            data_df.loc[task_col, hs_id] = metric_val
    return data_df

"""
    Computes contrast estimation based on medians in 4 steps as described in:
    Garcia et al. 2010 https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/sicidm/2010-Garcia-INS.pdf
    see pages 6-8
    exp 3.2
"""
def compute_custom_cem_exp3_2(results_df, iter_max, metric='total_hits'):
    # get data in dataset (rows) vs strategy (columns) format
    cem_data_df = get_stat_test_dict_exp3_2(results_df, iter_max, metric)

    # perform steps 1 and 2 of computing Zuv matrix
    num_algorithms = cem_data_df.columns.shape[0]
    algorithm_names = cem_data_df.columns.tolist()
    Zuv_matrix = pd.DataFrame(data=np.zeros(shape=(num_algorithms, num_algorithms)),
                              columns=algorithm_names,
                              index=algorithm_names)
    
    for u_idx in range(num_algorithms):
        for v_idx in range(u_idx+1, num_algorithms):
            u = algorithm_names[u_idx]
            v = algorithm_names[v_idx]
            
            tmp_df = cem_data_df[[u, v]].copy()
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

"""
    for exp 3.2
"""
def compute_scmamp_cem_exp3_2(results_df, iter_max, metric='total_hits'):
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from stat_analysis import setup_scmamp_rpy2
    setup_scmamp_rpy2()

    scmamp = rpackages.importr('scmamp')
    cem_data_df = get_stat_test_dict_exp3_2(results_df, iter_max, metric)
    cem_res = scmamp.contrastEstimationMatrix(cem_data_df)
    cem_df = pd.DataFrame(cem_res, columns=cem_data_df.columns, index=cem_data_df.columns)
    return cem_df

"""
    for exp 3.2
"""
def plot_cem_heatmap_exp3_2(results_df, task_info, metric, 
                             title, figsize=(16, 16), add_fail_success_counts=True,
                             shrink_factor=0.1, fontsize=45, title_y=1.05, 
                             hspace=0.2, wspace=0.2):
    
    from matplotlib.collections import QuadMesh
    from matplotlib.text import Text
    hs_ids_order = ['CBWS_341', 'CBWS_55', 'CBWS_609', 
                    'MABSelector_2', 'MABSelector_exploitive', 'CBWS_custom_1',
                    'ClusterBasedRandom', 'InstanceBasedRandom']
    iter_max_dict = {9010: 10, 9020: 20, 9030: 30, 9040: 40, 9050: 50}
    task_info = task_info.sort_values('active_ratio')
    tasks = task_info['task_col'].tolist()
    
    if add_fail_success_counts:
        fs_df_list, _, _ = get_exp3_2_failures(results_df, task_info)
        fs_df = fs_df_list[0]
            
    cem_wins_dict = {}
    #fig, axes = plt.subplots(2, len(iter_max_dict), figsize=figsize)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(2, 3, wspace=wspace, hspace=hspace)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    axes = [[ax1, ax2, ax3]]

    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1])
    axes.append([ax1, ax2, None])
    axes = np.array(axes).flatten()

    for axes_i, iter_max in enumerate(iter_max_dict):
        curr_title='# iterations: {}'.format(iter_max_dict[iter_max])
        
        cem_df = compute_custom_cem_exp3_2(results_df, iter_max, metric)
        
        heatmap_df = cem_df.copy()
        heatmap_df[' '] = 0
        heatmap_df['Total Wins'] = (cem_df > 0).sum(axis=1)
        heatmap_df = heatmap_df.loc[hs_ids_order]
        heatmap_df = heatmap_df[hs_ids_order + [' ', 'Total Wins']]
        facecolor_limit = 3

        if add_fail_success_counts:
            heatmap_df['# Failures (out of 102)'] = np.nan
            heatmap_df['# >= 10%'] = np.nan
            heatmap_df['# >= 25%'] = np.nan
            heatmap_df['# >= 50%'] = np.nan
            facecolor_limit=7
            for hs_id in hs_ids_order[:-2]:
                failures_cnt = fs_df_list[0][(fs_df_list[0].columns[0][0],iter_max_dict[iter_max])].loc[hs_id]
                a = fs_df_list[1][(fs_df_list[1].columns[0][0],iter_max_dict[iter_max])].loc[hs_id]
                b = fs_df_list[2][(fs_df_list[2].columns[0][0],iter_max_dict[iter_max])].loc[hs_id]
                c = fs_df_list[3][(fs_df_list[3].columns[0][0],iter_max_dict[iter_max])].loc[hs_id]
                heatmap_df.loc[hs_id, '# Failures (out of 102)'] = failures_cnt
                heatmap_df.loc[hs_id, '# >= 10%'] = a
                heatmap_df.loc[hs_id, '# >= 25%'] = b
                heatmap_df.loc[hs_id, '# >= 50%'] = c

        labels = []
        for i, row in heatmap_df.iterrows():
            x = int(row['Total Wins'])
            addendum_labels = ['', '{}'.format(x)]
            if add_fail_success_counts:
                f, a, b, c = row['# Failures (out of 102)'], row['# >= 10%'], row['# >= 25%'], row['# >= 50%'] 
                if not np.isnan(f):
                    f = int(f)
                if not np.isnan(a):
                    a = int(a)
                if not np.isnan(b):
                    b = int(b)
                if not np.isnan(c):
                    c = int(c)
                    
                addendum_labels += ['{}'.format(f), '{}'.format(a), '{}'.format(b), '{}'.format(c)]
            tmp = ['' for _ in range(heatmap_df.shape[0])] + addendum_labels
            labels.append(tmp)
        labels = np.array(labels)
        
        cmap = plt.get_cmap("RdYlGn")
        sns.heatmap(heatmap_df, annot=labels, linewidths=1, linecolor='grey', cmap=cmap,
                    fmt='', square=True, cbar_kws={"shrink": shrink_factor}, ax=axes[axes_i])

        # find your QuadMesh object and get array of colors
        quadmesh = axes[axes_i].findobj(QuadMesh)[0]
        facecolors = quadmesh.get_facecolors()

        # make colors of the last column white

        # set modified colors
        quadmesh.set_facecolors = facecolors

        for i in range(1, facecolor_limit):
            facecolors[np.arange(heatmap_df.shape[1]-i, heatmap_df.shape[0]*heatmap_df.shape[1], 
                                 heatmap_df.shape[1])] = np.array([1,1,1,1])
        
        locs = axes[axes_i].get_xticks()
        locs = [i+0.35 for i in locs]
        axes[axes_i].set_xticks(locs)
    
       # set color of all text to black
        for i in axes[axes_i].findobj(Text):
            i.set_color('black')

        axes[axes_i].set_title(curr_title, y=1.06, fontsize=fontsize)
        if axes_i%3 > 0:
            axes[axes_i].set_yticks([])
        
        if axes_i > 1:
            axes[axes_i].set_xticklabels(axes[axes_i].get_xticklabels(), rotation=70, ha='right')
            
        cem_wins_df = heatmap_df['Total Wins']
        cem_wins_dict[iter_max] = cem_wins_df
            
    plt.tight_layout();
    #plt.suptitle(title, fontsize=fontsize, y=title_y)
    plt.show();
    return cem_wins_dict