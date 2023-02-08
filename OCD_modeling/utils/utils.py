# Miscalenous utility functions

import platform

def get_working_dir():
    """ get computer name to set working path """
    if platform.node()=='qimr18844':
        working_dir = '/home/sebastin/working/'
    elif 'hpcnode' in platform.node():
        working_dir = '/mnt/lustre/working/'
    else:
        print('Computer unknown! Setting working dir as /working')
        working_dir = '/working/'
    return working_dir
