### Create new simulations from inference of posterior distributions
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime 
#import duckdb
import itertools
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os 
import pandas as pd
import pickle
import sqlite3
#import sqlalchemy as sl
from time import time, sleep

#import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils.utils import proj_dir, today
from OCD_modeling.mcmc.history_analysis import import_results, compute_kdes
from OCD_modeling.mcmc.abc_hpc import unpack_params, get_prior, get_prior_Thal, get_prior_Thal_fc_weak, get_prior_Thal_hc_weak, get_prior_scan_con, get_prior_scan_con_refined, get_prior_scan_con_2_nodes
from OCD_modeling.models.ReducedWongWang import create_sim_df
from OCD_modeling.hpc.parallel_launcher import run_sim

#from OCD_modeling.utils import proj_dir, today
#from OCD_modeling.mcmc import import_results, compute_kdes, unpack_params, get_prior
#from OCD_modeling.models import create_sim_df
#from OCD_modeling.hpc import run_sim

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch


priorfunc = {4:get_prior, 6:get_prior_Thal_hc_weak} #6:get_prior_Thal_fc_weak  #get_prior_Thal
priorfunc = {2:get_prior_scan_con_2_nodes, 4:get_prior_scan_con_refined, 6:get_prior_Thal_hc_weak} #6:get_prior_Thal_fc_weak  #get_prior_Thal

def create_params(kdes, cols, test_param, args):
    """ creates n_sims new parameters from posetrior distribution of base cohort """
    model_params, sim_params, control_params, bold_params, ids = [],[],[],[],[] # not most elegant but fine for now
    params = [] # to keep formatted parameters easier to convert to DataFrame
    _,bounds = priorfunc[args.N]()
    # create a new dict of params to unpack
    for i in range(args.n_sims):
        param = dict()
        # original param
        for col in cols:
            param[col] = kdes[args.base_cohort][col]['kde'].sample().squeeze()    
            # make sure it's within bounds
            while( (param[col]<bounds[col][0]) | (param[col]>bounds[col][1])):
                param[col] = kdes[args.base_cohort][col]['kde'].sample().squeeze()    
            
            # test parameters
            if col in test_param.split(' '):
                param[col] = kdes[args.test_cohort][col]['kde'].sample().squeeze()
                # make sure it's within bounds
                while( (param[col]<bounds[col][0]) | (param[col]>bounds[col][1])):
                    param[col] = kdes[args.test_cohort][col]['kde'].sample().squeeze()    
                
        model_pars, sim_pars, ctl_pars, bold_pars = unpack_params(param)
        
        # put those params in format digestable by pool.map
        model_params.append(model_pars)
        sim_params.append(sim_pars)
        control_params.append(ctl_pars)
        bold_params.append(bold_pars)
        params.append(param)
        ids.append(i)
    return model_params, sim_params, control_params, bold_params, params, ids


def use_optim_params(cols, test_param, args):
    """ Re-use parameter sets from outputs of the optimization """ 
    histories = import_results(args)
    model_params, sim_params, control_params, bold_params, paired_ids = [],[],[],[],[] # not most elegant but fine for now
    params = [] # to keep formatted parameters easier to convert to DataFrame

    base_particles = histories[args.base_cohort].get_population(t=9).particles
    test_particles = histories[args.test_cohort].get_population(t=9).particles
    n_particles = len(base_particles)
    
    # create a new dict of params to unpack
    for i in range(args.n_sims):
        param = dict()
        j = i % n_particles
        # original param
        for col in cols:
            param[col] = base_particles[j].parameter[col]    

            # test params
            if col in test_param.split(' '):
                param[col] = test_particles[j].parameter[col]
                
        model_pars, sim_pars, ctl_pars, bold_pars = unpack_params(param)

        # put those params in format digestable by pool.map
        model_params.append(model_pars)
        sim_params.append(sim_pars)
        control_params.append(ctl_pars)
        bold_params.append(bold_pars)
        params.append(param)
        paired_ids.append(j)
    return model_params, sim_params, control_params, bold_params, params, paired_ids


def write_outputs_to_db(params, cols, test_param, outputs, paired_ids, args):
    """ Write output of simulation inference into SQLite database """
    t = time()
    dbpath = os.path.join(proj_dir, 'postprocessing', args.db_name+'.db')
    with sqlite3.connect(dbpath, isolation_level='EXCLUSIVE', timeout=args.timeout) as conn:
    #engine = sl.create_engine("duckdb:///"+dbpath)
    #sl.MetaData().create_all(engine)
    #with engine.connect() as conn:
        # may need this to acquire exclusive lock from beginning
        #conn.execute("BEGIN IMMEDIATE")
        # if table not exitsing in db, it will create it at he end of the function
        cursor = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='SIMTEST'")
        if cursor.fetchall()[0][0]==0:
            cnt=0
            idx=0
        else:
            # get how many simulations in db
            cursor = conn.execute("SELECT COUNT(*) FROM SIMTEST WHERE base_cohort='{}'".format(args.base_cohort))
            cnt = cursor.fetchall()[0][0]
            cursor = conn.execute(" SELECT COUNT(*) FROM SIMTEST ")
            idx = cursor.fetchall()[0][0]

        # get FC from simulations
        df_sims = create_sim_df(outputs, sim_type='sim-'+args.base_cohort[:3], offset=cnt, dataset=args.dataset)
        df_sims = df_sims.pivot(index=['subj', 'cohort'], columns='pathway', values='corr').reset_index()
        df_sims['base_cohort'] = args.base_cohort
        df_sims['test_cohort'] = args.test_cohort
        df_sims['test_param'] = test_param
        df_sims['paired_ids'] = paired_ids

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
        df_sims.to_sql('SIMTEST', conn, if_exists='append', index=False, method='multi')
    conn.close()
    print("took {:.3f}s".format(time()-t))


def launch_sims_parallel(kdes, cols, test_param, args=None):
    """ Run batched simulations from posterior inference in parallel:
    
    Parameters
    ----------
        kdes: dict
            Kernel Density Estimates from optimization (structured as kdes[group][param])
        cols: list
            Parameters (columns) which draw samples from KDEs (otherwise default values are used)
        test_param: list
            Parameters for which the posterior is permuted for the virtual interventions.
        args: argparse.Namespace
            Extra arguments with options.

    Returns
    -------
        None. Output of the simulations are written into the local SQLite database.

    """

    if args.use_optim_params:
        model_params, sim_params, control_params, bold_params, params, paired_ids = use_optim_params(cols, test_param, args)
    else:
        model_params, sim_params, control_params, bold_params, params, paired_ids = create_params(kdes, cols, test_param, args)
    tot_batches = int(np.ceil(args.n_sims/args.n_batch))
    batch_number = 1
    print("Starting a pool of {} workers to run {} simulation(s) by batches of {} with test parameters {}".format(args.n_jobs, args.n_sims, args.n_batch, test_param))
    with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool: 
    #with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('fork')) as pool: 
    #with multiprocessing.pool.Pool(processes=args.n_jobs) as pool:
    
        for model_pars, sim_pars,control_pars, bold_pars, pars, p_ids in zip(batched(model_params, args.n_batch), 
                                                                        batched(sim_params, args.n_batch), 
                                                                        batched(control_params, args.n_batch), 
                                                                        batched(bold_params, args.n_batch), 
                                                                        batched(params, args.n_batch),
                                                                        batched(paired_ids, args.n_batch)):
            
            t = time()
            #print("Starting a pool of {} workers to run a batch of {} simulation(s) with test parameters {}".format(args.n_jobs, args.n_batch, test_params))
            #with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool: 
            sim_objs = pool.map(run_sim, 
                            model_pars, sim_pars, control_pars, bold_pars)
            #sim_objs = joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(OCD_modeling.hpc.parallel_launcher.run_sim)
            #                                               (model_pars[i], sim_pars[i], control_pars[i], bold_pars[i]) for i in range(args.n_batch))
            
            outputs = list(sim_objs)
            write_outputs_to_db(pars, cols, test_param, outputs, p_ids, args)
            print("    Batch {}/{} done for test_param {} in {:.2f}s.".format(batch_number, tot_batches, test_param, time()-t))

            if args.save_outputs:
                fname = "sim_objs"+today()+"_batch{:04d}".format(batch_number)+".pkl"
                with open(os.path.join(proj_dir, 'traces/tmp', fname), 'wb') as f:
                    pickle.dump(outputs,f)
                print("Batch saved in traces/tmp/"+fname)

            batch_number += 1
        
        
def get_test_param(args):
    """ either get test params from command line arguments or from file """
    if args.test_param_index==None:
        if args.test_param!=[]:
            test_param = ' '.join(args.test_param)
        else:
            test_param = 'None'
    else:
        with open(os.path.join(proj_dir, 'postprocessing', 'params_combinations.pkl'), 'rb') as f:
            combinations = pickle.load(f)
            test_param = list(combinations[args.test_param_index])
            test_param = ' '.join(test_param)
    return test_param


def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations in intervention (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--n_batch', type=int, default=10, action='store', help="number of sumluation per batch between saving to DB (to reduce i/o overhead)")
    parser.add_argument('--gens', nargs='+', default=[], action='store', help="generation of the optimization (list, must be same length as histories)")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--histories', nargs='+', default=['rww4D_OU_HPC_20230510', 'rww4D_OU_HPC_20230605'], action='store', help="optimizations to analyse and compare")
    parser.add_argument('--history_names', type=list, default=['controls', 'patients'], action='store', help="names given to each otpimization loaded")
    parser.add_argument('--save_kdes', default=False, action='store_true', help='save KDEs')
    parser.add_argument('--db_name', type=str, default=None, help="identifier of the sqlite3 database")
    parser.add_argument('--base_cohort', type=str, default='controls', help="Cohort from which to infer posterior as default")
    parser.add_argument('--test_cohort', type=str, default='patients', help="Cohort from which to infer posterior of individual params")
    parser.add_argument('--test_param', nargs='+', default=[], help="posterior parameter to swap between base and test cohort, if empty list then all params are tested")
    parser.add_argument('--test_param_index', type=int, default=None, action='store', help='if using a external file for test parameters, use line index ')
    parser.add_argument('--timeout', type=int, default=3600, help="timeout for DB writting in parallel")
    parser.add_argument('--use_optim_params', default=False, action='store_true', help="if flag is on, use parameters from accepted particles of optimization, other draw new params from posterior")
    parser.add_argument('--N', type=int, default=4, action='store', help="Number of regions in simulations (default=4, could also be 6.")
    parser.add_argument('--dataset', type=str, default='OCD_baseline', action='store', help="Dataset to compare to. default:OCD_baseline. Other possible:OCD_SCAN_CON")
    args = parser.parse_args()
    args.gens = np.array(args.gens, dtype=int)
    return args


if __name__=='__main__':
    args = parse_arguments()
    # load histories and KDEs
    histories = import_results(args)
    kdes,cols = compute_kdes(histories, args=args)

    test_param = get_test_param(args)

    # delay start randomly by up to 3 min to avoid large batches of simulations writting
    # concurrently to DB 
    #sleep(np.random.randint(0,180))
    
    launch_sims_parallel(kdes, cols, test_param, args=args)

