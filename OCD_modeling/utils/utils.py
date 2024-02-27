# Miscalenous utility functions / common imports

import argparse
from datetime import datetime
import numpy as np
import os 
import platform
import scipy

def cohen_d(x,y):
    """ Calculates effect size as cohen's d """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def emd(u,v):
    """ computes the Wasserstein distance (i.e. earth mover's distance) across pathways P between u and v """
    d = []
    for col in u.columns:
        d.append(scipy.stats.wasserstein_distance(u[col], v[col]))
    return np.sum(d)


def rmse(u,v):
    """ compute the root mean squared error of correlation accross pathways P between u and v as 
    :math:`d = \sqrt{ \sum_{p \in P} (\mu_u^p - \mu_v^p)^2 + (\sigma_u_p - \sigma_v^p)^2}` 
    
    Parameters:
    -----------
    u,v
        pandas DataFrames with only pathway columns

    Returns:
    --------
    d
        Root Mean Squared Error 
    """
    u_ = u.apply([np.mean, np.std])
    v_ = v.apply([np.mean, np.std])
    mse = u_.combine(v_, np.subtract).apply('square').apply('sum').sum()
    return np.sqrt(mse)


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
        print('Computer unknown! Setting working dir as /working')
        working_dir = '/working/'
    return working_dir

working_dir = get_working_dir()
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')
