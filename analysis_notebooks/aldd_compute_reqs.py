"""
    Usage:
        python aldd_compute_reqs.py \
        --log_file_name=../logs/aldd.log \
        --job_name=HS_ClusterBasedRandom_TASK_pcba-aid602332_BATCH_0_START_1_ITERS_0 \
        --condor_subtemplate=./aldd_template.sub \
        --new_condor_subname=./HS_ClusterBasedRandom_TASK_pcba-aid602332_BATCH_0_START_1_ITERS_1.sub
"""

import argparse
import os

if __name__ ==  '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', action="store", dest="log_file_name", required=True)
    parser.add_argument('--job_name', action="store", dest="job_name", required=True)
    parser.add_argument('--condor_subtemplate', action="store", dest="condor_subtemplate", required=True)
    parser.add_argument('--new_condor_subname', action="store", dest="new_condor_subname", required=True)
    
    given_args = parser.parse_args()
    log_file_name = given_args.log_file_name
    job_name = given_args.job_name
    condor_subtemplate = given_args.condor_subtemplate
    new_condor_subname = given_args.new_condor_subname
    
    # static vars
    num_termlines = 18
    kb_to_gb = 1e-6
    mb_to_gb = 1e-3
    # increment in GB
    disk_increment = 0.02
    ram_increment = 0.1
    disk_max = 30
    ram_max = 12
    disk_min = 25
    ram_min = 4
    
    try:
        # read in log content
        with open(log_file_name, 'r') as f:
            content = f.readlines()

        # get job id assigned by condor
        line_idx = [i for i, x in enumerate(content) if job_name in x][0]
        job_id = content[line_idx-1].split('(')[1].split(')')[0]

        # get job termination output and usage
        job_term = [i for i, x in enumerate(content) if (job_id in x) & ('terminated' in x)][0]
        job_term = content[job_term:job_term+num_termlines]
        disk_usage = [i for i, x in enumerate(job_term) if 'Disk' in x][0]
        ram_usage = [i for i, x in enumerate(job_term) if 'Memory' in x][0]

        disk_usage = int(job_term[disk_usage].split()[3])
        ram_usage = int(job_term[ram_usage].split()[3])

        # add increment 
        disk_usage *= kb_to_gb 
        disk_usage += disk_increment

        disk_usage = min(disk_usage, disk_max)
        disk_usage = max(disk_usage, disk_min)
        disk_reqs = round(disk_usage, 2)
        
        ram_usage *= mb_to_gb 
        ram_usage += ram_increment    
        ram_usage = min(ram_usage, ram_max)
        ram_usage = max(ram_usage, ram_min)
        ram_reqs = round(ram_usage, 2)

        del content
        
        # update template sub file and save
        with open(condor_subtemplate, 'r') as f:
            subcontent = f.readlines()

        # find disk and ram reqs line
        disk_line = [i for i, x in enumerate(subcontent) if 'Request_disk ' in x][0]
        ram_line = [i for i, x in enumerate(subcontent) if 'Request_memory ' in x][0]

        subcontent[disk_line] = 'Request_disk  = {}GB\n'.format(disk_reqs)
        subcontent[ram_line] = 'Request_memory = {}GB\n'.format(ram_reqs)

        # write new submit file
        with open(new_condor_subname, 'w') as f:
            f.writelines(subcontent)
    except:
        # update template sub file and save
        with open(condor_subtemplate, 'r') as f:
            subcontent = f.readlines()

        # write new submit file
        with open(new_condor_subname, 'w') as f:
            f.writelines(subcontent)