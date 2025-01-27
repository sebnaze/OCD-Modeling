# Analysis of the modified Reduced Wong-Wang model
# incorporating Ornstein-Uhlenbeck process (OU) for 
# striato-cortical projections.
# 
# Note: This is tailored for 2-nodes version (N=2) 
# 
# Sebastien Naze
# QIMR Berghofer 2024

import argparse
import copy
import dill
import importlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sbn
import sys
import time

import OCD_modeling
from OCD_modeling.models import ReducedWongWang as RWW
from OCD_modeling.utils.utils import *


def postprocessing(sims, args):
    """ Extract transition times and FC to avoid saving whole simulation object """ 
    out = []
    for sim in sims:
        output = dict()
        output['eta'] = sim.eta_C[0,1]
        output['sigma'] = sim.sigma_C[0,1]
        output['fc'] = sim.bold_fc[0,1]
        output['n_transitions'] = sim.strFr_stats['C_12']['n_transitions']
        out.append(output)

    if args.save_outputs:
        os.makedirs(os.path.join(proj_dir, 'postprocessing', args.batch_id), exist_ok=True)
        with open(os.path.join(proj_dir, 'postprocessing', args.batch_id, str(args.param_index)+'.pkl'), 'wb') as f:
            pickle.dump(out, f)
    
    return output


def get_params(args):
    """ either get params from command line arguments or from file """
    if args.param_index==None:
            return {}
    with open(os.path.join(proj_dir, 'postprocessing', 'eta_sigma_params.pkl'), 'rb') as f:
            combinations = pickle.load(f)
            params = combinations[args.param_index]
            params.n_jobs = args.n_jobs
    return params


def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--param_index', type=int, default=None, action='store', help='if using a external file for parameters, use line index ')
    parser.add_argument('--batch_id', type=str, default='tmp', action='store', help='where to save simulations outputs')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    params = get_params(args)
    sims = OCD_modeling.hpc.launch_pool_simulations(params)
    output = postprocessing(sims, args)