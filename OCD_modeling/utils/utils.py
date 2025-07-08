# Miscalenous utility functions / common imports

import argparse
from datetime import datetime
import numpy as np
import os 
import platform
import scipy
import tomli


def cohen_d(x,y):
    """ Calculates effect size as cohen's d """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def paired_euclidian(u,v):
    """ Euclidian distance between paired simulations """ 
    u,v = np.array(u), np.array(v)
    return np.sqrt(np.sum((u - v)**2, axis=1))


def emd(u,v):
    """ computes the Wasserstein distance (i.e. earth mover's distance) across pathways P between u and v """
    d = []
    for col in u.columns:
        d.append(scipy.stats.wasserstein_distance(u[col], v[col]))
    return np.sum(d)


def rmse(u,v):
    """ compute the root mean squared error of correlation accross pathways P between u and v as 
    :math:`d = \sqrt{ \sum_{p \in P} (\mu_u^p - \mu_v^p)^2}` 
    
    Parameters:
    -----------
    u,v
        pandas DataFrames with only pathway columns

    Returns:
    --------
    d
        Root Mean Squared Error 
    """
    u_ = u.apply(np.mean)
    v_ = v.apply(np.mean)
    mse = u_.combine(v_, np.subtract).apply('square').sum()
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
    elif 'hpc' in platform.node():
        working_dir = '/mnt/lustre/working/'
    else:
        #print('Setting working dir as /working')
        working_dir = '/data/working/'
    return working_dir


def read_config(conf_file):
    """ Open and read TOML configuration file """
    #with open(conf_filename, 'rb') as f:
    config = tomli.load(conf_file)
    global proj_dir
    proj_dir = config['utils']['proj_dir']
    os.makedirs(os.path.join(proj_dir, 'traces'), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, 'postprocessing'), exist_ok=True)
    return config


working_dir = get_working_dir()

# default proj_dir if no config file is given
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')
