import numpy as np
import pandas as pd
import os
import glob
import shutil
from IPython.display import clear_output

def get_top_summary(all_df, summary_iter_num=9999):
    def _helper_agg(col):
        if col.name == 'rf_id':
            return '-'
        elif col.name in ['hs_id', 'hs_group']:
            return col.unique()[0]
        else:
            if col.name == 'run_counts':
                return col.sum()
            if '_std' in col.name:
                return col.std()
            else:
                return col.mean()
        
    temp_df = all_df[all_df['iter_num'] == summary_iter_num].drop('iter_num', axis=1)
    orig_col_ord = temp_df.columns.tolist()
    temp_df.columns = [x+'_mean'  if '_hits' in x else x for x in temp_df.columns]
    
    for c in [x for x in temp_df.columns if '_mean' in x]:
        temp_df[c.replace('_mean', '_std')] = temp_df[c]
        
    temp_df['run_counts'] = 1
    
    top_df = temp_df.groupby('hs_id').agg(_helper_agg)
    top_df = top_df.sort_values('total_hits_mean', ascending=False)
    
    des_order = ['exploitation_hits_mean', 'exploration_hits_mean', 'total_hits_mean', 'total_hits_std',
                 'exploitation_unique_hits_mean', 'exploration_unique_hits_mean','total_unique_hits_mean', 'total_unique_hits_std',
                 'exploitation_batch_size', 'exploration_batch_size', 'total_batch_size']
    return top_df

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
    job_df = pd.concat([job_df, job_df.sum().to_frame().T])
    job_df['iter_num'] = list(np.arange(iter_max)) + [9999]
    job_df['max_iter'] = job_df['iter_num'].sort_values().iloc[-2]
    
    return job_df

def get_results(results_dir, iter_max=10, task_col='pcba-aid624173', cluster_col='BT_0.4 ID', run_count_threshold=5,
                check_failure=True, drop_runs=True, isExp3=False):
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
        
        if isExp3:
            task_col = rd_splits[3]
            rf_id = rd_splits[4]
            batch_size = rd_splits[5]
        else:
            rf_id = rd_splits[3]
            batch_size = rd_splits[4]

        # check that the job completed succesfully:
        # - exactly iter_max*batch_size cpds were selected and that they have unique Index ID
        batch_cpds = glob.glob(rdir+'iter_*/expl*.csv')
        if len(batch_cpds) > 0:
            cpd_df = pd.concat([pd.read_csv(x) for x in batch_cpds])
            if cpd_df['Index ID'].unique().shape[0] < iter_max*int(batch_size.split('_')[-1]):
                print('Failed at {}'.format(hs_id))
            if check_failure:
                if cpd_df.shape[0] == iter_max*int(batch_size.split('_')[-1]):
                    successful_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
                    assert cpd_df['Index ID'].unique().shape[0] == iter_max*int(batch_size.split('_')[-1])
                else:
                    failed_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
                    continue
            else:
                if cpd_df['Index ID'].unique().shape[0] == cpd_df.shape[0]:
                    successful_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
                else:
                    failed_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
                    continue
                
        else:
            failed_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
            continue
        
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
    
    def _drop_runs(all_df):
        all_df = all_df[all_df['iter_num'] == 9999]
        u_hs_id, run_counts = np.unique(all_df['hs_id'], return_counts=True)
        all_df = all_df[all_df['hs_id'].isin(u_hs_id[np.where(run_counts > run_count_threshold)[0]])]
        return all_df
        
    if len(all_96) > 0:
        all_96 = pd.concat(all_96)
        if drop_runs:
            all_96 = _drop_runs(all_96)
    else:
        all_96 = None
        
    if len(all_384) > 0:
        all_384 = pd.concat(all_384)
        if drop_runs:
            all_384 = _drop_runs(all_384)
    else:
        all_384 = None
    
    if len(all_1536) > 0:
        all_1536 = pd.concat(all_1536)
        if drop_runs:
            all_1536 = _drop_runs(all_1536)
    else:
        all_1536 = None
    all_df = pd.concat([all_96, all_384, all_1536])
    
    return all_96, all_384, all_1536, all_df, successful_jobs, failed_jobs

def get_results_old():
    successful_jobs = []
    failed_jobs = []

    all_0 = []
    all_1 = []
    all_2 = []
    for i, sf in enumerate(summary_files):
        clear_output()
        print('{}/{}'.format(i, len(summary_files)))
        sf_splits = sf.split('\\')
        hs_group = sf_splits[1]
        hs_id = sf_splits[2]
        rf_id = sf_splits[3]
        batch_size = sf_splits[4]

        batch_cpds = glob.glob(sf+'iter_*\expl*.csv')
        if len(batch_cpds) > 0:
            cpd_df = pd.concat([pd.read_csv(x) for x in batch_cpds])

            if cpd_df.shape[0] == 10*int(batch_size.split('_')[-1]):
                assert cpd_df['Index ID'].unique().shape[0] == 10*int(batch_size.split('_')[-1])
                successful_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
            else:
                failed_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
                continue
        else:
            failed_jobs.append('{}_rf_{}_{}'.format(hs_id, rf_id, batch_size))
            continue

        des_cols = ['iter_num', 
                    'exploitation_hits', 'exploitation_max_hits', 'exploitation_batch_size',
                    'exploration_hits', 'exploration_max_hits', 'exploration_batch_size',
                    'total_hits', 'total_max_hits', 'total_batch_size']
        summary_df = pd.read_csv(sf+'\summary.csv')
        summary_df = summary_df[des_cols]
        summary_df['rf_id'] = rf_id
        summary_df['hs_id'] = hs_id
        summary_df['batch_size'] = int(batch_size.split('_')[-1])
        summary_df['hs_group'] = hs_group

        if int(batch_size.split('_')[-1]) == 96:
            all_0.append(summary_df)
        elif int(batch_size.split('_')[-1]) == 384:
            all_1.append(summary_df)
        else:
            all_2.append(summary_df)

    all_0 = pd.concat(all_0)
    all_1 = pd.concat(all_1)
    all_2 = pd.concat(all_2)
    all_df = pd.concat([all_0, all_1, all_2])