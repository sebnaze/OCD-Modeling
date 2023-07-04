### Analyse inferred simulations w.r.t. observed data
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os 
import pandas as pd
import pickle
import pdb
import pyabc
import scipy
import seaborn as sbn
import sklearn
import sqlite3 as sl

import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils.utils import *
from OCD_modeling.mcmc.history_analysis import import_results, compute_kdes
from OCD_modeling.mcmc.simulate_inference import batched


def load_df_sims(args):
    """ Load infered simulations from database """
    with sl.connect(os.path.join(proj_dir, 'postprocessing', args.db_name+'.db')) as conn:
        df = pd.read_sql(''' SELECT * FROM SIMTEST ''', conn)
    conn.close()
    return df

def load_df_data(args):
    """ Loads clinical FC data in pandas Dataframe """
    with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_avg_2023.pkl'), 'rb') as f:
        df_data = pickle.load(f)
    return df_data


def compute_dist(in_tuple):
    patient, sim_vecs, df_data, pathways, args = in_tuple
    """ compute distance between single patient to each simulation """
    pat_vec = np.array([df_data[(df_data.subj==patient)&(df_data.pathway==p)]['corr'] for p in pathways])
    sims = []
    for sim_name,sim_vec in sim_vecs:
        d = np.sqrt(np.sum((a-b)**2 for a,b in zip(pat_vec,sim_vec)))
        if d < args.tolerance:
            sims.append({'sim':sim_name, 'distance':d})
    return sims


def compute_distances(df_data, df_sims, args):
    """ Compute euclidian distance between single empirical FC and simulated FC """ 
    print("Computing distances between patients and simulations")
    pathways = np.sort(df_data.pathway.unique())
    #params = np.sort([par for par in df_sims.test_param.unique() if not par.contains('None')])
    #nrows = args.n_sims*len(pathways)
    #df_sim_coh = df_sims[(df_sims.test_param=='None')&(df_sims.base_cohort==args.base_cohort)]
    patients = np.sort(df_data[df_data.cohort=='patients'].subj.unique())
    #sim_names = np.sort(df_sim_coh.subj.unique())
    
    # prepare distance vectors of simulations
    #sim_vecs = []
    #for sim_name in sim_names:
    #    sim_vec = [df_sim_coh[(df_sim_coh.subj==sim_name)&(df_sim_coh.pathway==p)]['corr'] for p in pathways]
    #    sim_vecs.append((sim_name, np.array(sim_vec)))

    def get_sim_vector(sim):
        sim_vec = np.array(sim[pathways])
        return (sim['subj'],sim_vec)
    
    print("extracting simulation vectors...")
    sim_vecs = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(get_sim_vector)(sim) for i,sim in df_sims.iterrows())

    print("computing distances in parallel...")
    assoc = dict()
    with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool: 
        pt_sims = pool.map(compute_dist, [(patient, sim_vecs, df_data, pathways, args) for patient in patients])
        pt_sims = list(pt_sims)
    for patient, sims in zip(patients, pt_sims):
        assoc[patient] = sims

    if args.save_distances:
        fname = os.path.join(proj_dir, 'postprocessing', args.db_name+'_distances100eps'+str(int(args.tolerance*100)+".pkl"))
        with open(fname, 'wb') as f:
            pickle.dump(assoc, f)
    return assoc


def merge_data_sim_dfs(df_pat, df_sims, assoc, args):
    """ Create single dataframe concatenating all patients relevant variables """
    dfs = []
    for pat in df_pat.subj.unique():
        if pat in assoc.keys():
            if len(assoc[pat]):
                df_sim_pat = pd.DataFrame(assoc[pat]).sort_values('distance')
                df_sim_pat['subj'] = pat
                df_sim_pat = df_pat[df_pat.subj==pat].merge(df_sim_pat)
                # keep shortest distance only (can be changed to X closest...)
                if df_sim_pat.iloc[0]['distance']<args.tolerance_plot:
                    df_sim = df_sims[df_sims.subj==df_sim_pat.iloc[0]['sim']]
                    dfs.append(df_sim_pat.iloc[0].to_frame().transpose().merge(df_sim.rename(columns={'subj':'sim'})))
                if args.verbose:
                    print('{}: merged data and sim.'.format(pat))
    return pd.concat(dfs, ignore_index=True)
 

def plot_param_behav(df_sim_pat, params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42'], behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total'], args=None):
    """ Plot association between simulation parameters and behavioral/clinical measures """
    for behav in behavs:
        fig = plt.figure(figsize=[24,4])
        for i,param in enumerate(params):
            ax = plt.subplot(1,len(params),i+1)

            sbn.scatterplot(data=df_sim_pat, x=behav, y=param, ax=ax)

            r,p = scipy.stats.pearsonr(df_sim_pat[behav], df_sim_pat[param])
            plt.title("r={:.2f}    p={:.3f}".format(r,p))
        plt.tight_layout()
        plt.show(block=False)
                

def fix_df_sims_names(df_sims, args):
    """ Fix duplicate sim names in dataframe (sometimes sqlite3 locking mechanism does fail, the db is not corrupted but 
      simulation indices may need fixing (duplicate names)  """
    df_sims['subj'] = df_sims.apply(lambda row: "sim-"+args.base_cohort[:3]+"{:06d}".format(int(row.name)+1), axis=1)



def multivariate_analysis():
    """ Multivariate regression of parameters to predict behaviors """


def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=192, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--n_batch', type=int, default=10, action='store', help="batch size")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--save_kdes', default=False, action='store_true', help='save KDEs')
    parser.add_argument('--db_name', type=str, default="sim_test_20230614", help="identifier of the sqlite3 database")
    parser.add_argument('--df_sims', type=str, default="df_sims_pat_20230621", help="identifier of the fixed dataframe")
    parser.add_argument('--base_cohort', type=str, default='controls', help="Cohort from which to infer posterior as default")
    parser.add_argument('--test_cohort', type=str, default='patients', help="Cohort from which to infer posterior of individual params")
    parser.add_argument('--test_params', nargs='+', default=[], help="posterior parameter to swap between base and test cohort, if empty list then all params are tested")
    parser.add_argument('--tolerance', type=float, default=0.05, action='store', help="maximal distance allow to take into consideration 'digital sigling'")
    parser.add_argument('--tolerance_plot', type=float, default=0.05, action='store', help="maximal distance allow to take into consideration 'digital sigling' (for plotting) ")
    parser.add_argument('--save_distances', default=False, action='store_true', help='save distances between patients and simulations')
    parser.add_argument('--load_distances', default=False, action='store_true', help='load distances between patients and simulations from previously saved')
    parser.add_argument('--plot_param_behav', default=False, action='store_true', help='plot param-behavioral relationship')
    parser.add_argument('--verbose', default=False, action='store_true', help='print extra processing info')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    # load histories and KDEs
    #histories = import_results(args)
    
    df_sims = load_df_sims(args)
    fix_df_sims_names(df_sims, args)
    df_data = load_df_data(args)

    if args.load_distances:
        fname = os.path.join(proj_dir, 'postprocessing', args.db_name+'_distances100eps'+str(int(args.tolerance*100))+".pkl")
        with open(fname, 'rb') as f:
            assoc = pickle.load(f)
    else:
        assoc = compute_distances(df_data, df_sims, args)

    with open(os.path.join(proj_dir, 'postprocessing', 'df_pat.pkl'), 'rb') as f:
        df_pat = pickle.load(f)

    df_sim_pat = merge_data_sim_dfs(df_pat, df_sims, assoc, args)
    if args.plot_param_behav:
        plot_param_behav(df_sim_pat, args=args)

    
