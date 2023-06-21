### Analyze history of optimizations
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
from matplotlib import pyplot as plt
import numpy as np
import os 
import pandas as pd
import pickle
import pyabc
import scipy
import sklearn

import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils.utils import *

def import_results(args):
    """ Read optimization results from DB """ 
    histories = dict()
    for i,db_name in enumerate(args.histories):
        db_path = os.path.join(proj_dir, 'traces', db_name+'.db')
        name = args.history_names[i]
        try:
            histories[name] = pyabc.History("sqlite:///"+db_path)
        except:
            print(f'Optimization {db_name} does not exist, check name.')
    return histories
    

def load_empirical_FC(args):
    """ Loads functional connectivity from patients and controls (data, not simulation) """
    with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_avg_2023.pkl'), 'rb') as f:
        df_data = pickle.load(f)
    return df_data       

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
            plt.subplot(2,6,j+1)
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
            n_cols = len(list(df.columns))
            for k,col in enumerate(df.columns):
                plt.subplot(n_gen+1,n_cols,j*n_cols+k+1)
                plt.hist(df[col], alpha=0.2)
                plt.xlabel(col)

                # use first generation (prior) to set xlims
                if j==0:
                    xs[col] = [df[col].min(), df[col].max()]
                plt.xlim(xs[col])
    plt.tight_layout()


def plot_kde_matrix(histories, args):
    for (name,history) in histories.items():
        df,w = history.get_distribution(m=0)
        pyabc.visualization.plot_kde_matrix(df, w)


def compute_stats(histories, args=None):
    """ Computes the statistics of the optimization outcome, i.e. tests the posterior distributions (parameters) 
    between controls and patients.
            returns: df_stats: pandas DataFrame of statistics  """
    df_post_con,w_con = histories['controls'].get_distribution(t=9)
    df_post_pat,w_pat = histories['patients'].get_distribution(t=9)
    cols = df_post_con.columns
    mc = len(cols)
    stats = []
    for col in cols:
        x,y = df_post_con[col], df_post_pat[col]
        
        stat_norm_con,p_norm_con = scipy.stats.normaltest(x)
        stat_norm_pat,p_norm_pat = scipy.stats.normaltest(y)

        t,p_t = scipy.stats.ttest_ind(x,y)
    
        u,p_u = scipy.stats.mannwhitneyu(x,y)
        
        h,p_h = scipy.stats.kruskal(x,y)
        
        line = "{:15}    t={:8.2f}    p={:.4f}    p_bf={:.4f}    normality(con/pat)={}/{}    U={:8d}    p={:.4f}    p_bf={:.4f}    H={:8.3f}    p={:.4f}    p_bf={:.4f}".format(col,t,p,p*mc,p_norm_con>0.05,p_norm_pat>0.05,int(u),p_u, p_u*mc,h,p_h,p_h*mc)
        print(line)
        
        df_line = {'param':col, 't':t, 'p_t':p_t, 'p_t_bf':p_t/len(cols), 'normality':(p_norm_con>0.05)&(p_norm_pat>0.05), \
                'u':u, 'p_u':p_u, 'p_u_bf':p_u/len(cols), \
                'h':h, 'p_h':p_h, 'p_h_bf':p_h/len(cols),}
        stats.append(df_line)
    df_stats = pd.DataFrame(stats)
    return df_stats


def compute_kdes(histories, n_pts = 100, args=None):
    """ 
    Computes Kernel Density Estimates (KDEs) of the posterior distributions of parameters.
    
        Inputs:
        -------
            histories (dict): nested dictionnary of SQL alchemy history objects
            n_pts (int): number of points used to estimate the probability density functions (PDFs)

        Outputs:
        --------
            kdes (dict): nested dictiorany of KDEs and associated PDFs
            cols (list): list of parameters for which the KDEs were estimated 
    """

    kdes = dict()

    cols = []
    for cohort,history in histories.items():
        kdes[cohort] = dict()
        df,w = history.get_distribution(t=9)
        for col in df.columns:
            cols.append(col)
            vmin, vmax = df[col].min(), df[col].max()
            bw = (vmax-vmin)/15
            kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(df[col]).reshape(-1,1))
            X = np.linspace(vmin-3*bw, vmax+3*bw, 100).reshape(-1,1)
            pdf = kde.score_samples(X)
            kdes[cohort][col] = {'kde':kde, 'pdf':pdf, 'X':X}
    cols = np.unique(cols)

    if args.save_kdes:
        fname = os.path.join()

    return kdes, cols


def plot_kdes(kdes, cols, args=None):
    """  Plot Kernel Density Estimates of posteriors """
    plt.figure(figsize=[20,6])
    for i,col in enumerate(cols):
        plt.subplot(2,7,i+1)
        plt.plot(kdes['controls'][col]['X'], kdes['controls'][col]['pdf'], 'lightblue')
        plt.plot(kdes['patients'][col]['X'], kdes['patients'][col]['pdf'], 'orange')
        plt.title(col)
    plt.tight_layout()
    if args.save_figs:
        today = datetime.now().strftime('_%Y%m%d')
        fname = os.path.join(proj_dir, 'img', 'kdes'+today+'.svg')
        plt.savefig(fname)
    plt.show()
    

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
    parser.add_argument('--save_kdes_suffix', type=str, default=today(), help="identifier of the KDE in the saving folder")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    
    histories = import_results(args)
    
    if args.plot_epsilons:
        plot_epsilons(histories, args)
    if args.plot_weights:
        plot_weights(histories, args)
    if args.plot_param_distrib:
        plot_param_distrib(histories, args)
    if args.plot_kde_matrix:
        plot_kde_matrix(histories, args)

    if args.plot_stats:
        df_stats = compute_stats(histories, args)
    
    if args.compute_kdes:
        kdes,cols = compute_kdes(histories, args)
    if args.plot_kdes:
        plot_kdes(kdes, cols, args)
