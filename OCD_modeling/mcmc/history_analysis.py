import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils.utils import *

import argparse
from matplotlib import pyplot as plt
import numpy as np
import os 
import pyabc

working_dir = get_working_dir()
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')

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
    args = parser.parse_args()
    return args


def import_results(args):
    """ Read optimization results from DB """ 
    histories = dict()
    for i,db_name in enumerate(args.histories):
        db_path = os.path.join(proj_dir, 'traces', db_name+'.db')
        name = args.history_names[i]
        try:
            histories[name] = pyabc.History("sqlite:///"+db_path)
        except:
            print(f'Optimization {db_name} does nor exist, check name.')
    return histories
    

def plot_epsilons(histories, args):
    """ Plot evolution of epsilons across generations """ 
    n_hist = len(histories)
    fig = plt.figure(figsize=[5, 5])
    ax = plt.subplot(1,1,1)
    for i,(name,history) in enumerate(histories.items()):
        pyabc.visualization.plot_epsilons([history], ax=ax)
    plt.legend(list(histories.keys()))


def plot_weights(histories, args):
    """ Plot evolution of weights across generations """ 
    n_hist = len(histories)
    n_gen = np.max([h.max_t for h in histories.values()])
    fig = plt.figure(figsize=[20, 5])
    for i,(name,history) in enumerate(histories.items()):
        for j in range(history.max_t+1):
            df,w  = history.get_distribution(t=j)
            plt.subplot(2,5,j+1)
            plt.scatter(np.arange(len(w)), w, s=10, alpha=0.5)
            plt.title(f"t={j}")
            plt.xlabel('particle')
            plt.ylabel('weight')
            #plt.legend(args.history_names)
    plt.tight_layout()


def plot_param_distrib(histories, args):
    """ plot posterior distribution at end of optimization """
    n_hist = len(histories)
    n_gen = np.max([h.max_t for h in histories.values()])
    fig = plt.figure(figsize=[40, 20])
    xs = dict() # xlims 
    for i,(name,history) in enumerate(histories.items()):
        for j in range(history.max_t+1):
            df,w  = history.get_distribution(t=j)
            for k,col in enumerate(df.columns):
                plt.subplot(10,14,j*14+k+1)
                plt.hist(df[col], alpha=0.2)
                plt.xlabel(col)

                # use first generation (prior) to set xlims
                if j==0:
                    xs[col] = [df[col].min(), df[col].max()]
                plt.xlim(xs[col])
    plt.tight_layout()


if __name__=='__main__':
    args = parse_arguments()
    
    histories = import_results(args)
    plot_epsilons(histories, args)
    plot_weights(histories, args)
    plot_param_distrib(histories, args)
