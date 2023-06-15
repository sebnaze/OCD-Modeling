###     Parallel simulations launched on HPC or local machine 
##
##      Author: Sebastien Naze
#
#       QIMR Berghofer 2023

import argparse 
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
from datetime import datetime
import joblib
import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle

from OCD_modeling.models import ReducedWongWang as RWW
from OCD_modeling.utils.utils import *

def run_sim(model_pars, sim_pars, control_pars={}, bold_pars={}):
    """ Run a single simulation """
    # run simulation 
    rww_sim = RWW.ReducedWongWangOU(**model_pars)
    rww_sim.set_control_params(control_pars)
    rww_sim.run(**sim_pars)

    # analyze traces
    RWW.compute_bold(rww_sim, **bold_pars)

    return rww_sim


def launch_simulations(args):
    """ Launch N simulations in parallel """
    sim_objs = joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(run_sim)(args.model_pars, args.sim_pars, args.control_pars, args.bold_pars) for _ in range(args.n_sims))
    return sim_objs


def launch_pool_simulations(args):
    """ Launch N simulations in parallel using a Pool Executor """
    #sim_objs = []
    with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        #print(pool._mp_context)
        #futures = {pool.submit(run_sim, args.model_pars, args.sim_pars, args.control_pars, args.bold_pars) : i for i in range(args.n_sims)}
        #for future in as_completed(futures):
        #    sim_objs.append(future.result())

        sim_objs = pool.map(run_sim, 
                                np.repeat(args.model_pars, args.n_sims), 
                                np.repeat(args.sim_pars, args.n_sims), 
                                np.repeat(args.control_pars, args.n_sims), 
                                np.repeat(args.bold_pars, args.n_sims))
    
    #for out in outs:
    #    sim_objs.append(*out)
        
    return list(sim_objs)
    

def save_batch(sim_objs, args):
    """ Save simulation runs as objects list in pickle file """
    if args.save_outputs:
        today = datetime.today().strftime('%Y%m%d')
        with open(os.path.join(proj_dir, 'postprocessing', 'sim_objs_'+today+'.pkl'), 'wb') as f:
            pickle.dump(sim_objs, f)


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
  parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
  parser.add_argument('--save_scores', default=False, action='store_true', help='save scores')
  parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
  parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
  parser.add_argument('--params_idx', type=int, default=0, action='store', help="index of the parameter set")
  parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
  parser.add_argument('--batch_id', type=str, default="test", action='store', help='batch number unique to each batched job launched on cluster, typically YYYYMMDDD_hhmmss')
  parser.add_argument('--model_pars', type=json.loads, default={}, action='store', help="dictionary of model parameters")
  parser.add_argument('--sim_pars', type=json.loads, default={}, action='store', help="dictionary of simulation (run) parameters")
  parser.add_argument('--control_pars', type=json.loads, default={}, action='store', help="dictionary of control parameters")
  parser.add_argument('--bold_pars', type=json.loads, default={}, action='store', help="dictionary of BOLD recording parameters")
  args = parser.parse_args()
  return args

if __name__=='__main__':
  args = parse_arguments()
  sim_objs = launch_pool_simulations(args)
  save_batch(sim_objs, args)