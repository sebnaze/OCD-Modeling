# Miscalenous utility functions

import platform

def get_working_dir():
    """ get computer name to set working path """
    if platform.node()=='qimr18844':
        working_dir = '/home/sebastin/working/'
    elif platform.node()=='neurosrv01':
        working_dir = '/home/sebastienn/working/'
    elif 'hpcnode' in platform.node():
        working_dir = '/mnt/lustre/working/'
    else:
        print('Mince! Computer unknown! Setting working dir as /working')
        working_dir = '/working/'
    return working_dir
