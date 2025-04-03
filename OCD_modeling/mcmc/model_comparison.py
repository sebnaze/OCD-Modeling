### Model comparison for the inference of OCD frontostriatal dysregulations.
#
# This analysis can be run after optimizations have been performed using different
# models, e.g. with different connectivity layout or priors. It compares how well 
# the model does to reproduce empirical functional connecticity in OCD subjects.

import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle 
import seaborn as sbn
import sqlite3 

from OCD_modeling.utils.utils import *

def load_empirical_data():
    """ Load empirical dataset that we optimized against. 
    
    Returns
    -------
        df_data: pandas.DataFrame
            Empirical dataset with subjects and functional connecticity (Pearson's correlation) in each 
            frontostriatal pathways. 
    """ 
    with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_avg_2023.pkl'), 'rb') as f:
        df_data = pickle.load(f)
    return df_data


def load_simulations(args):
    """ Load simulated inferences.
    
    Parameters
    ----------
        args: argparse.Namespace
            Dictionnary-like datastructure containing ``args.db_names`` argument which was enterred
            in command-line as a list of simulated dataset.

    Returns
    -------
        out: list
            List of ``pandas.DataFrame``, each containing *n* simulations with functional connectivity 
            in each frontostriatal pathway. (default n=1000)  

    """
    out = []
    for db_name in args.db_names:
        with sqlite3.connect(os.path.join(proj_dir, 'postprocessing', db_name)) as conn:
            df = pd.read_sql(''' SELECT * FROM SIMTEST ''', conn)
            out.append(df)
        conn.close()
    return out
 
def compute_errors(df_sims, df_data, args):
    """ Compute root-mean-square errors between empirical and simulated data in frontostriatal pathways 
    functional connectivity.

    This error takes the form of a distribution, i.e. with 1000 simulations grouped 
    by cohorts of 50, it makes 20 error values.

    Parameters
    ----------
        df_sims: pandas.DataFrame
            Simulated dataset.
        df_data: pandas.DataFrame
            Empirical dataset.
        args: argparse.Namespace
            Optional arguments.
    
    Returns
    -------
        pandas.DataFrame
            Distribution of errors in functinal connectivity space.
    """
    df_data_formatted = df_data.pivot(index=['subj', 'cohort'], columns='pathway', values='corr').reset_index()
    n_sim = 50
    errs = []
    for i in np.arange(0,args.n_sims, n_sim):
        df_ = df_sims.iloc[i:i+n_sim]
        err = []
        for pathway in ['Acc_OFC', 'dPut_PFC']:
        #for pathway in df_data.pathway.unique(): #['Acc_OFC', 'dPut_PFC']:
            mu_data = df_data_formatted[df_data_formatted.cohort=='patients'][pathway].mean()
            sigma_data = df_data_formatted[df_data_formatted.cohort=='patients'][pathway].std()
            mu_sim = df_[pathway].mean()
            sigma_sim = df_[pathway].std()
            
            err.append([np.power(mu_data-mu_sim,2), np.power(sigma_data-sigma_sim,2)])
        err_ = np.concatenate(err)
        errs.append(np.sqrt(err_.sum()/len(err_)))
    return pd.DataFrame({'error': np.array(errs)})


def plot_errors(df_errors, args):
    """ Box plot of inference error on frontostriatal functional connectivity 
    
    Parameters
    ----------
        df_errors: pandas.DataFrame
            Distributions of errors for each models to compare.
        args: argparse.Namespace
            Optional arguments, including ``args.model_tags`` that contains labels of model names for saving.
    
    """ 
    plt.figure(figsize=[np.ceil(len(args.db_names)/2),2])
    ax = sbn.boxplot(data=df_errors, x='model', y='error', width=0.6, boxprops={'alpha':0.6}, linewidth=1, fliersize=0)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    lbls = ax.get_xticklabels()
    ax.set_xticklabels(lbls, rotation=30)

    if args.save_figs:
        fname = os.path.join(proj_dir, 'img', 'model_comparison_'+'_'.join(args.model_tags)+today()+'.svg')
        plt.savefig(fname)

    plt.show()

def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=1000, action='store', help="number of simulations in intervention (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--db_names', type=str, nargs='+', default=[], help="identifier of the sqlite3 databases where inferences are stored")
    parser.add_argument('--model_tags', type=str, nargs='+', default=[], help="model tags for labeling each assessed model")
    parser.add_argument('--N', type=int, default=4, action='store', help="Number of regions in simulations (default=4, could also be 6.")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    
    df_data = load_empirical_data()

    sims = load_simulations(args)

    df_errors = []
    for i,df_sims in enumerate(sims):
        df_error = compute_errors(df_sims, df_data, args)
        df_error['model'] = args.model_tags[i]
        df_errors.append(df_error)
    df_errors = pd.concat(df_errors)

    plot_errors(df_errors, args)