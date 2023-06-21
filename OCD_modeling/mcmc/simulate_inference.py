### Create new simulations from inference of posterior distributions
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor
import itertools
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os 
import pandas as pd
import pdb
import pyabc
import scipy
import sklearn
import sqlite3 as sl
from time import time

import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils.utils import *
from OCD_modeling.mcmc.history_analysis import import_results, compute_kdes
from OCD_modeling.mcmc.abc_hpc import default_bold_params, default_model_params, default_sim_params, unpack_params
from OCD_modeling.models.ReducedWongWang import create_sim_df

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch

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
    dbpath = os.path.join(proj_dir, 'postprocessing', args.db_name+'.db')
    with sl.connect(dbpath, isolation_level='EXCLUSIVE') as conn:
        # if table not exitsing in db, it will create it at he end of the function
        cursor = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='SIMTEST'")
        if cursor.fetchall()[0][0]==0:
            cnt=0
            idx=0
        else:
            # get how many simulations in db
            cursor = conn.execute(" SELECT COUNT(*) FROM SIMTEST WHERE base_cohort IS '{}'".format(args.base_cohort))
            cnt = cursor.fetchall()[0][0]
            cursor = conn.execute(" SELECT COUNT(*) FROM SIMTEST ")
            idx = cursor.fetchall()[0][0]

        # get FC from simulations
        df_sims = create_sim_df(outputs, sim_type='sim-'+args.base_cohort[:3], offset=cnt)
        df_sims = df_sims.pivot(index=['subj', 'cohort'], columns='pathway', values='corr').reset_index()
        df_sims['base_cohort'] = args.base_cohort
        df_sims['test_cohort'] = args.test_cohort
        df_sims['test_param'] = test_param

        # get associated parameters
        df_params = pd.DataFrame.from_dict(params, dtype=float)
        df_sims = df_sims.join(df_params)
        #pars = dict((col,[]) for col in cols)
        #for param in params:
        #    for par,val in param.items():
        #        pars[par].append(val)
        #for col,vals in pars.items():
        #    df_sims[col] = np.repeat(vals,6) # create_sim_df return a dataframe with 6 rows per simulation

        # save in DB
        df_sims.to_sql('SIMTEST', conn, if_exists='append', index=False)
    conn.close()


def launch_sims_parallel(kdes, cols, args):
    """ Start simulations from posterior inference in parallel """
    if args.test_params==[]:
        args.test_params = cols
    
    for test_param in args.test_params:
        model_params, sim_params, control_params, bold_params, params = create_params(kdes, cols, test_param, args)
        tot_batches = int(np.ceil(args.n_sims/args.n_batch))
        batch_number = 1
        print("Starting a pool of {} workers to run {} simulation(s) by batches of {} with test parameter {}".format(args.n_jobs, args.n_sims, args.n_batch, test_param))
        with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool: 
            for model_pars, sim_pars,control_pars, bold_pars, pars in zip(batched(model_params, args.n_batch), 
                                                                          batched(sim_params, args.n_batch), 
                                                                          batched(control_params, args.n_batch), 
                                                                          batched(bold_params, args.n_batch), 
                                                                          batched(params, args.n_batch)):
                t = time()
                sim_objs = pool.map(OCD_modeling.hpc.parallel_launcher.run_sim, 
                                    model_pars, sim_pars, control_pars, bold_pars)
                outputs = list(sim_objs)
                write_outputs_to_db(pars, cols, test_param, outputs, args)
                print("    Batch {}/{} done for test_param {} in {:.2f}s.".format(batch_number, tot_batches, test_param, time()-t))
                batch_number += 1
        #outputs = []
        #for mp,sp,cp,bp in zip(model_pars, sim_pars, control_pars, bold_pars):
        #    outputs.append(OCD_modeling.hpc.parallel_launcher.run_sim(mp,sp,cp,bp))
        
        

def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--n_batch', type=int, default=10, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
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

