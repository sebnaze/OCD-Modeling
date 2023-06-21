# Miscalenous utility functions / common imports

import argparse
from datetime import datetime
import os 
import platform

def today():
    return datetime.now().strftime('_%Y%m%d')

def get_working_dir():
    """ get computer name to set working path """
    if platform.node()=='qimr18844':
        working_dir = '/home/sebastin/working/'
    elif platform.node()=='qimr17596':
        working_dir = '/home/sebastin/working/'
    elif platform.node()=='neurosrv01':
        working_dir = '/home/sebastienn/working/'
    elif 'hpcnode' in platform.node():
        working_dir = '/mnt/lustre/working/'
    else:
        print('Mince! Computer unknown! Setting working dir as /working')
        working_dir = '/working/'
    return working_dir

working_dir = get_working_dir()
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')
