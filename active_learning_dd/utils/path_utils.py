"""
    Contains path and path formatting utils. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

    
"""
    Returns common results directory path from a given config.
"""
def get_common_results_path(config, iter_num):
    common_results_path = config['paths']['params_set_results_dir'] + \
                          config['paths']['iter_results_dir'].format(iter_num)
    return common_results_path
    
    
"""
    Returns common eval file path from a given config. 
"""
def get_eval_file_path(config, iter_num):
    return get_common_results_path(config, iter_num) + config['paths']['eval_dest_file']
    