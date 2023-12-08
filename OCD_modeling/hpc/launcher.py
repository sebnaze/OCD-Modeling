# Launcher of the OCD model on hpc
#
# Author: Sebastien Naze
#
# QIMR Berghofer 2022-2023

import argparse
import glob
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import platform
import seaborn as sbn
import sys
import time

import OCD_modeling
from OCD_modeling.hpc import launch_simulations
from OCD_modeling.models import ReducedWongWang as RWW
from OCD_modeling.utils.utils import *

def load_params(args):
    """ load parameters set based on input index """
    with open(os.path.join(proj_dir, 'log', args.batch_id, args.param_file), 'rb') as f:
        params = pickle.load(f)
    return params


def param_exploration(args):
    print("Batch {}".format(args.batch_id))
    t0 = time.time()
    p = load_params(args)[args.params_idx]
    C = np.array([  [0,p['C_12'],p['C_13'],0], # to OFC
                    [p['C_21'],0,0,p['C_24']], # to PFC
                    [p['C_31'],0,0,p['C_34']], # to NAcc
                    [0,p['C_42'],p['C_43'],0]  # to Put
                ])
    rww = RWW.ReducedWongWangND(dt=0.01, N=4, C=C, G=p['G'], sigma=p['sigma'])
    print("Running model #{:08d}".format(args.params_idx))
    rww.run(t_tot=1000, sf=100, t_rec=[100,1000])
    RWW.compute_bold(rww)
    r, corr_diff, corr = RWW.score_model(rww)
    score =  {'r': r, 'corr_diff':corr_diff, 'corr':corr}

    if args.save_scores:
        print("Saving score for model #{:08d}".format(args.params_idx))
        fpath = os.path.join(proj_dir, 'log', args.batch_id, 'paramID_{:08d}'.format(args.params_idx))
        os.makedirs(fpath, exist_ok=True)
        with open(os.path.join(fpath, 'score.pkl'), 'wb') as f:
            pickle.dump(score, f)
        with open(os.path.join(fpath, 'param.txt'), 'w') as f:
            f.write(json.dumps(p))
    print("Done in {:.2f}s.".format(time.time()-t0))


def set_params(p):
    """ Setup default simulation parameters """
    opts = argparse.Namespace()
    opts.model_pars = {'N':2, 'G':2.3, 'C':np.array([[0,0.3], [0.2,0]]), 'sigma':0.06, 
                       'eta_C':np.array([[0,p['eta_c']],[0,0]]), 'sigma_C':np.array([[0,p['sigma_c']],[0,0]])}
    opts.control_pars = {}
    opts.sim_pars = {'t_tot': 36000, 't_rec':[500, 36000]}
    opts.bold_pars = {}
    opts.n_sims = 100
    return opts


def save_transitions(output, args):
    """ extract state transitions from simulated traces """
    transitions = {'eta_c':output['eta_c'], 'sigma_c':output['sigma_c'], 'S1_transitions':[], 'S2_transitions':[]}
    for model in output['sim_objs']:
        transitions['S1_transitions'].append(model.transitions['S1'])
        transitions['S2_transitions'].append(model.transitions['S1'])
        
    with open(os.path.join(proj_dir, 'traces', args.batch_id, 'transitions_paramID_{:08d}.pkl'.format(args.params_idx)), 'wb') as f:
        pickle.dump(transitions, f)
    

def compute_transitions(args):
    """ Run batches of n_sims simulations to show etaC and sigmaC effect on transition probability """
    print("Batch {}".format(args.batch_id))
    t0 = time.time()
    p = load_params(args)[args.params_idx]
    opts = set_params(p)
    opts.n_jobs = args.n_jobs
    
    print('Launching {} simulations with eta_c={} and sigma_c={}'.format(opts.n_sims, p['eta_c'], p['sigma_c']))
    sim_objs = launch_simulations(args=opts)
    output = {'eta_c':p['eta_c'], 'sigma_c':p['sigma_c'], 'sim_objs':sim_objs}
    print('Done in {}s'.format(int(time.time()-t0)))

    if args.save_outputs:
        os.makedirs(os.path.join(proj_dir, 'traces', args.batch_id), exist_ok=True)
        save_transitions(output, args)


def analyze_transitions(args):
    """ Analyze output from the transitions simulations """
    files = glob.glob(os.path.join(proj_dir, 'traces', args.batch_id, 'transitions_paramID_*'))
    print('Analyzing transitions from {} parameter configurations'.format(len(files)))
    df_lines = [] 
    for file in files:
        with open(file, 'rb') as f:
            transitions = pickle.load(f)
            for trs in transitions['S1_transitions']:
                df_lines.append({'eta_c':transitions['eta_c'], 'sigma_c':transitions['sigma_c'], 
                                 'variable':'S1', 'n_transitions':len(trs)})
            for trs in transitions['S2_transitions']:
                df_lines.append({'eta_c':transitions['eta_c'], 'sigma_c':transitions['sigma_c'], 
                                 'variable':'S2', 'n_transitions':len(trs)})
    df_transitions = pd.DataFrame(df_lines)
    return df_transitions


def plot_transitions(df_transitions, args):
    """ Plot transition rates """
    df_transitions['n_transitions'] = df_transitions['n_transitions']/9.86
    sbn.barplot(data=df_transitions, x=['eta_c', 'sigma_c'], y='n_transitions', hue='variable', hue_order=['S1', 'S2'])
    plt.ylabel('transition rate (/hr)')


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
  parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
  parser.add_argument('--save_scores', default=False, action='store_true', help='save scores')
  parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
  parser.add_argument('--param_file', type=str, default="params.pkl", action='store', help='file stroring parameters')
  parser.add_argument('--params_idx', type=int, default=0, action='store', help="index of the parameter set")
  parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
  parser.add_argument('--batch_id', type=str, default="test", action='store', help='batch number unique to each batched job launched on cluster, typically YYYYMMDDD_hhmmss')
  parser.add_argument('--param_exploration', default=False, action='store_true', help='Run brute force parameter exporation (deprecated -- used early in project before using Bayesian optimization)')
  parser.add_argument('--compute_transitions', default=False, action='store_true', help='Run simulation(s) to extarct transition probablities')
  parser.add_argument('--analyze_transitions', default=False, action='store_true', help='Analyse transition probabilities')
  args = parser.parse_args()
  return args

if __name__=='__main__':
    args = parse_arguments()
    if args.param_exploration:
        param_exploration(args)

    if args.compute_transitions:
        compute_transitions(args)

    if args.analyze_transitions:
        df_transitions = analyze_transitions(args)
        if args.plot_figs:
            plot_transitions(df_transitions, args)
