### Create new simulations from inference of posterior distributions
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os 
import pandas as pd
import pyabc
import scipy
import sklearn
import sqlite3 as sl

import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils.utils import *
from OCD_modeling.mcmc.history_analysis import import_results, compute_kdes
from OCD_modeling.mcmc.abc_hpc import default_bold_params, default_model_params, default_sim_params, unpack_params
from OCD_modeling.models.ReducedWongWang import create_sim_df

def create_params(kdes, cols, test_param, args):
    """ creates n_sims new parameters from posetrior distribution of base cohort """
    model_params, sim_params, control_params, bold_params = [],[],[],[] # not most elegant but fine for now
    params = [] # to keep formatted parameters easier to convert to DataFrame
    # create a new dict of params to unpack
    for i in range(args.n_sims):
        param = dict()
        for col in cols:
            param[col] = kdes[args.base_cohort][col]['kde'].sample().squeeze()
            if col==test_param:
                param[col] = kdes[args.test_cohort][test_param]['kde'].sample().squeeze()
        model_pars, sim_pars, ctl_pars, bold_pars = unpack_params(param)
        
        # put those params in format digestable by pool.map
        model_params.append(model_pars)
        sim_params.append(sim_pars)
        control_params.append(ctl_pars)
        bold_params.append(bold_pars)
        params.append(param)
    return model_params, sim_params, control_params, bold_params, params 


def write_outputs_to_db(params, cols, test_param, outputs, args):
    """ Write output of simulation inference into SQLite database """
    # get FC from simulations
    df_sims = create_sim_df(outputs, sim_type='sim-test')
    df_sims['base_cohort'] = args.base_cohort
    df_sims['test_cohort'] = args.test_cohort
    df_sims['test_param'] = test_param

    # get associated parameters
    pars = dict((col,[]) for col in cols)
    for param in params:
        for par,val in param.items():
            pars[par].append(val)
    for col,vals in pars.items():
        df_sims[col] = np.repeat(vals,6) # create_sim_df return a dataframe with 6 rows per simulation

    # save in DB
    with sl.connect(os.path.join(proj_dir, 'postprocessing', args.db_name+'.db')) as conn:
        df_sims.to_sql('SIMTEST', conn, if_exists='append')


def launch_sims_parallel(kdes, cols, args):
    """ Start simulations from posterior inference in parallel """
    if args.test_params==[]:
        args.test_params = cols
    
    for test_param in args.test_params:
        model_pars, sim_pars, control_pars, bold_pars, params = create_params(kdes, cols, test_param, args)
        
        print("Starting a pool of {} workers to run {} simulation(s) with test parameter {}".format(args.n_jobs, args.n_sims, test_param))
        with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool: 
            sim_objs = pool.map(OCD_modeling.hpc.parallel_launcher.run_sim, 
                                model_pars, sim_pars, control_pars, bold_pars)
        outputs = list(sim_objs)
        #outputs = []
        #for mp,sp,cp,bp in zip(model_pars, sim_pars, control_pars, bold_pars):
        #    outputs.append(OCD_modeling.hpc.parallel_launcher.run_sim(mp,sp,cp,bp))
        write_outputs_to_db(params, cols, test_param, outputs, args)
        print("Done with {}, saved to DB".format(test_param))
        

def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--gens', type=list, default=[], action='store', help="generation of the optimization (list, must be same length as histories)")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--histories', nargs='+', default=None, action='store', help="optimizations to analyse and compare")
    parser.add_argument('--history_names', type=list, default=['controls', 'patients'], action='store', help="names given to each otpimization loaded")
    parser.add_argument('--save_kdes', default=False, action='store_true', help='save KDEs')
    parser.add_argument('--db_name', type=str, default=None, help="identifier of the sqlite3 database")
    parser.add_argument('--base_cohort', type=str, default='controls', help="Cohort from which to infer posterior as default")
    parser.add_argument('--test_cohort', type=str, default='patients', help="Cohort from which to infer posterior of individual params")
    parser.add_argument('--test_params', nargs='+', default=[], help="posterior parameter to swap between base and test cohort, if empty list then all params are tested")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    # load histories and KDEs
    histories = import_results(args)
    kdes,cols = compute_kdes(histories, args=args)

    launch_sims_parallel(kdes, cols, args)

