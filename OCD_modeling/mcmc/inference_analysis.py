### Analyse inferred simulations w.r.t. observed data
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
from joblib import Parallel, delayed
import matplotlib as mpl
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os 
import pandas as pd
import pickle
import pingouin as pg
import pyabc
import scipy
import seaborn as sbn
import sklearn
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit, LeaveOneOut
from sklearn.metrics import make_scorer, r2_score
from sklearn.inspection import permutation_importance
import sqlite3 as sl
from statsmodels.stats.weightstats import ztest

# import most relevant environment and project variable
from OCD_modeling.utils.utils import proj_dir, today, rmse, emd, cohen_d, paired_euclidian
from OCD_modeling.mcmc.history_analysis import import_results, compute_kdes
from OCD_modeling.analysis.fc_data_analysis import drop_single_session

# mapping of parameter names from numbered to lettered indices (e.g. C_13 to C_OA)
param_mapping = {'C_12':'C_OL', 'C_13':'C_OA', 'C_21':'C_LO', 'C_24':'C_LP', 'C_31':'C_AO', 'C_34':'C_AP', 'C_42':'C_PL', 'C_43':'C_PA',
                 'eta_C_13':'eta_OA', 'eta_C_24':'eta_LP', 'sigma_C_13':'sigma_OA', 'sigma_C_24':'sigma_LP', 'sigma':'sigma', 'G':'G',
                 'patients':'patients', 'controls':'controls'}

metric = {'rmse': rmse, 'emd': emd}

def load_df_sims(args):
    """ Load infered simulations from database """
    if type(args.db_names)==list:
        dfs = []
        for db_name in args.db_names:
            with sl.connect(os.path.join(proj_dir, 'postprocessing', db_name+'.db')) as conn:
                df = pd.read_sql(''' SELECT * FROM SIMTEST ''', conn)
            conn.close()
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True, copy=False)
    else:
        with sl.connect(os.path.join(proj_dir, 'postprocessing', args.db_names+'.db')) as conn:
            df = pd.read_sql(''' SELECT * FROM SIMTEST ''', conn)
        conn.close()
        return df

def load_df_data(args):
    """ Loads clinical FC data in pandas Dataframe """
    with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_avg_2023.pkl'), 'rb') as f:
        df_data = pickle.load(f)
    return df_data


def load_kdes(args):
    """ Load Kernel Density Estimates from Optimization outcomes (see history_analysis.py) """
    #fname = 'kdes_rww4D_OU_HPC_20230510_rww4D_OU_HPC_20230605_20230919.pkl' # <-- compatible scikit-learn v0.24.3
    fname = 'kdes_rww4D_OU_HPC_20230510_rww4D_OU_HPC_20230605_20240417.pkl'  # <-- compatible scikit-learn v1.1.3
    with open(os.path.join(proj_dir, 'postprocessing', fname), 'rb') as f:
        kdes = pickle.load(f)
    return kdes


def load_dist_to_FC_controls(args):
    """ Load distances in FC space of data (control & patients) to controls from clinical trial (PRE and POST) """ 
    #fname = os.path.join(proj_dir, 'postprocessing', 'distances_to_FC_controls_20230907.pkl')
    #fname = os.path.join(proj_dir, 'postprocessing', 'distances_to_FC_controls_20240305.pkl')
    fname = os.path.join(proj_dir, 'postprocessing', 'distances_to_FC_controls_20240418.pkl')
    with open(fname, 'rb') as f:
        distances = pickle.load(f)
        df_fc_pre_post = distances['indiv']
        df_fc_pre_post = df_fc_pre_post.reset_index(drop=True)
        df_fc_pre_post.index.name = None
    return df_fc_pre_post


#-------------------------#
#   Digital twin analysis #
#-------------------------#

def get_sim_vector(sim):
        sim_vec = np.array(sim[pathways])
        return (sim['subj'],sim_vec)


def compute_sim_vecs(df_sims):
    """ Compute vector of each simulation in FC space in parallel """
    sim_vecs = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(get_sim_vector)(sim) for i,sim in df_sims.iterrows())
    return sim_vecs



def compute_distances(df_data, df_sims, ses, args):
    """ Compute Euclidian distances between single empirical functional connectivity (FC) and simulated FC. 
    
    Parameters
    ----------
        df_data: pandas.DataFrame
            Empirical FC.
        df_sims: pandas.DataFrame
            Simulated FC.
        ses: string
            Session (i.e. point in time): initial ("ses-pre") or follow-up ("ses-post") appointment.
        args: argparse.Namespace
            Extra arguments with options.

    Returns
    -------
        assoc: dict
            Unique pairing between individual OCD subjects and simulated subjects (digital twins).
         
    """ 
    print("Computing distances between patients and simulations")
    pathways = np.sort(df_data.pathway.unique())
    patients = np.sort(df_data[df_data.cohort=='patients'].subj.unique())

    sim_vecs = df_sims[pathways]
    sim_vecs = zip(df_sims['subj'], sim_vecs)

    def compute_dist(patient, df_data, pathways, args):
        """ compute distance between single patient to each simulation """
        pat_vec = np.array([df_data[(df_data.subj==patient)&(df_data.pathway==p)]['corr'] for p in pathways])
        sims = []
        for _,sim_row in df_sims.iterrows():
            sim_vec = np.array(sim_row[pathways])
            sim_name = sim_row['subj']
            d = np.sqrt(sum((a-b)**2 for a,b in zip(pat_vec,sim_vec)))[0]
            if d < args.tolerance:
                sims.append({'sim':sim_name, 'distance':d})
        return sims

    assoc = dict()
    #with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool: 
    #with ProcessPoolExecutor(max_workers=args.n_jobs) as pool: 
    #    pt_sims = pool.map(compute_dist, [(patient, sim_vecs, df_data, pathways, args) for patient in patients])
    #    pt_sims = list(pt_sims)
    pt_sims = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(compute_dist)(patient, df_data, pathways, args) for patient in patients)
    for patient, sims in zip(patients, pt_sims):
        assoc[patient] = sims

    if args.save_distances:
        if len(args.db_names)==1:
            fname = os.path.join(proj_dir, 'postprocessing', args.db_names[0]+'_distances100eps'+str(int(args.tolerance*100))+"_"+ses+today()+".pkl")
        else:
            fname = os.path.join(proj_dir, 'postprocessing', 'assoc_digital_twins_distances100eps'+str(int(args.tolerance*100))+"_"+ses+today()+".pkl")
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
                # keep n-th distance entry only (can be changed to n-th closest...)
                for i,sim_row in df_sim_pat.iloc[:args.n_closest].iterrows():
                    if sim_row['distance'] < args.tolerance_plot:
                        df_sim = df_sims[df_sims.subj==sim_row['sim']]
                        dfs.append(sim_row.to_frame().transpose().merge(df_sim.rename(columns={'subj':'sim'})))
                if args.verbose:
                    print('{}: merged data and sim.'.format(pat))
    return pd.concat(dfs, ignore_index=True)
 

def plot_param_behav(df_sim_pat, params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42'], behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total'], args=None):
    """ Plot association between simulation parameters and behavioral/clinical measures """
    for behav in behavs:
        fig = plt.figure(figsize=[4*len(params),4])
        for i,param in enumerate(params):
            ax = plt.subplot(1,len(params),i+1)

            sbn.scatterplot(data=df_sim_pat, x=behav, y=param, ax=ax)

            r,p = scipy.stats.pearsonr(df_sim_pat[behav], df_sim_pat[param])
            plt.title("r={:.2f}    p={:.3f}".format(r,p))
        plt.tight_layout()
        plt.show(block=False)
                

#---------------------------#
#   Multivariate Analysis   #
#---------------------------#

def fix_df_sims_names(df_sims, args):
    """ Fix duplicate sim names in dataframe (sometimes sqlite3 locking mechanism does fail, the db is not corrupted but 
      simulation indices may need fixing (duplicate names)  """
    df_sims['subj'] = df_sims.apply(lambda row: "sim{:06d}".format(int(row.name)+1), axis=1)
    df_sims['n_test_params'] = df_sims.test_param.apply(lambda pars_str: 0 if pars_str=='none' else len(pars_str.split(' ')))



def score_pval(y, y_pred):
    """ scoring based on correlation p-value """
    res = scipy.stats.pearsonr(y, y_pred)
    return res.pvalue


def cross_validation(model, X, y, args):
    """ Run cross validation using CV type given in argument """
    # home made p-value scorer
    p_score = make_scorer(score_pval, greater_is_better=False)


    cv_types = {'RepeatedKFold': {'cv': RepeatedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats), 
                                  'scoring': {'p':p_score, 'r2':make_scorer(r2_score)}},
                'KFold': {'cv': KFold(n_splits=args.n_splits),
                          'scoring': {'p':p_score, 'r2':make_scorer(r2_score)}},
                'ShuffleSplit': {'cv': ShuffleSplit(n_splits=args.n_splits, test_size=args.test_size),
                                 'scoring': {'p':p_score, 'r2':make_scorer(r2_score)}},
                'LeaveOneOut': {'cv': LeaveOneOut(),
                                'scoring': 'neg_mean_absolute_error'}
                }

    cv_results = sklearn.model_selection.cross_validate(
            copy.deepcopy(model),
            X,
            y,
            cv=cv_types[args.cv_type]['cv'],
            scoring=cv_types[args.cv_type]['scoring'],
            return_estimator=True,
            return_train_score=True,
            return_indices=True,
            n_jobs=args.n_jobs,
    )
    return cv_results


def multivariate_analysis(df_sim_pat, params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42'], 
                          behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total', 'Anx_total', 'Dep_Total'], 
                          models={'Ridge':sklearn.linear_model.Ridge(alpha=0.01)}, null=False, args=None):
    """ Multivariate regression of parameters to predict behaviors """
    output = dict()
    for i,behav in enumerate(behavs):
        output[behav] = dict()
        # Linear regression
        X = df_sim_pat[params] 
        y = df_sim_pat[behav]
            
        for model_name,model in models.items():
            # create null couplings 
            if null:
                # Null based on full matrix permutation 
                #data=np.array(df_sim_pat[params])
                #data = np.random.permutation(data.ravel()).reshape(data.shape)
                #X = pd.DataFrame(data=data, columns=params)

                # Null based on column-wise permutations 
                data=np.array([np.random.permutation(df_sim_pat[param]) for param in params]).T
                X = pd.DataFrame(data=data, columns=params)
            model.fit(X,y)
            y_pred = model.predict(X)
            mae = sklearn.metrics.median_absolute_error(y, y_pred)
            print("Median Abs Error (MAE) = {:.5f}".format(mae))

            r2 = sklearn.metrics.r2_score(y, y_pred)

            # stats on prediction 
            r,p = scipy.stats.pearsonr(y,y_pred)

            # cross validation
            cv_results = cross_validation(model, X, y, args)            

            output[behav][model_name] = {'model':copy.deepcopy(model), 'X':X, 'y':y, 'y_pred':y_pred, 'r2':r2, 'r':r, 'p':p, 
                                         'cv_results':copy.deepcopy(cv_results)}
    return output

def plot_multivariate_results(multivar, models=['Ridge'], args=None):
    """ Display results of the multivariate analysis """
    behavs = multivar.keys()
    for model_name in models:
        fig = plt.figure(figsize=[3*len(behavs),12])
        gs = plt.GridSpec(4,len(behavs))
        for i,behav in enumerate(behavs):

            y, y_pred = multivar[behav][model_name]['y'], multivar[behav][model_name]['y_pred']
            r2, r, p = multivar[behav][model_name]['r2'], multivar[behav][model_name]['r'], multivar[behav][model_name]['p']
            model = multivar[behav][model_name]['model']
            X = multivar[behav][model_name]['X']

            # PLOTTING 
            # -------- 
            # linear regression
            ax = fig.add_subplot(gs[0,i])
            ax.scatter(y, y_pred, alpha=0.3, color='k')
            ax.set_xlabel(behav)
            ax.set_ylabel("Prediction")
            ax.set_title("r2={:.2f} r={:.2} p={:.3}".format(r2,r,p))

            # coefficients
            ax = fig.add_subplot(gs[1,i])
            coefs = pd.DataFrame(data=[model.coef_], columns=model.feature_names_in_)
            sbn.barplot(data=coefs, orient='h', alpha=0.5, ax=ax)
            plt.title('coefs')
                        
            # normalized coefficients
            ax = fig.add_subplot(gs[2,i])
            coefs = pd.DataFrame(data=[model.coef_ * np.std(X)], columns=model.feature_names_in_)
            sbn.barplot(data=coefs, orient='h', alpha=0.5, ax=ax)
            plt.title('normalized coefs')
            
            if model_name=='Ridge':
                var_coefs = pd.DataFrame(
                    [ est.coef_ for est in multivar[behav][model_name]['cv_results']['estimator'] ], 
                    columns=model.feature_names_in_ )    
                
                # coefficients' variability
                ax = fig.add_subplot(gs[3,i])
                sbn.stripplot(data=var_coefs, orient="h", palette="dark:k", alpha=0.5)
                sbn.boxplot(data=var_coefs, orient="h", saturation=0.7, whis=10)
                plt.axvline(x=0, color=".5")
                plt.xlabel("Coefficients")
                plt.title("Coefficients' variability")
                    
        plt.tight_layout()
        plt.show()


def plot_cv_regression(multivar, df_sim_pat, args=None):
    """ Scatter plots of cross-validated regression """
    behavs = multivar.keys()
    params = multivar['YBOCS_Total']['Ridge']['model'].feature_names_in_
    for behav in behavs:
        train_ids = multivar[behav]['Ridge']['cv_results']['indices']['train']
        test_ids = multivar[behav]['Ridge']['cv_results']['indices']['test']

        # Plot training and test in each fold
        n_folds = len(train_ids)
        n_rows = int(np.ceil(n_folds/5))
        plt.figure(figsize=[15,n_rows*3])
        tests = {'y':[], 'y_pred':[]}
        for i,train_idx in enumerate(train_ids):
            test_idx = test_ids[i]
            est = multivar[behav]['Ridge']['cv_results']['estimator'][i]

            X_train = df_sim_pat.iloc[train_idx][params]
            X_test = df_sim_pat.iloc[test_idx][params]
            y_train = df_sim_pat.iloc[train_idx][behav]
            y_test = df_sim_pat.iloc[test_idx][behav]
            train_pred = est.predict(X_train)
            test_pred = est.predict(X_test)

            tests['y'].append(y_test)
            tests['y_pred'].append(test_pred)

            plt.subplot(n_rows,5,i+1)
            plt.scatter(y_train, train_pred, label='train')
            plt.scatter(y_test, test_pred, label='test')
            plt.xlabel(behav+' ground truth')
            plt.ylabel(behav+' prediction')

            r2 = sklearn.metrics.r2_score(y_train, train_pred)
            r,p = scipy.stats.pearsonr(y_train, train_pred)
            plt.title("training\nr2={:.2f}  r={:.2f}  p={:.3f}".format(r2,r,p))
        plt.tight_layout()
        plt.show(block=False)

        # Plot all tests aggregated over folds
        plt.figure(figsize=[4,3])
        y = np.array(tests['y']).ravel().squeeze()
        y_pred = np.array(tests['y_pred']).ravel().squeeze()
        r, p = scipy.stats.pearsonr(y, y_pred)
        r2 = sklearn.metrics.r2_score(y, y_pred)
        plt.scatter(y, y_pred, color='orange')
        plt.xlabel(behav+' ground truth')
        plt.ylabel(behav+' prediction')
        plt.title("testing\nr2={:.2f}  r={:.2f}  p={:.3f}".format(r2,r,p)); 
        plt.tight_layout()
        plt.show(block=False)



def plot_multivar_svd(multivar, behavs=None, args=None):
    """ dimensionality reduction on linear regression parameters/coefficients """
    behavs = list(multivar.keys()) if behavs==None else behavs
    coefs = []
    for behav in behavs:
        coefs.append(multivar[behav]['Ridge']['model'].coef_)
        params = multivar[behav]['Ridge']['model'].feature_names_in_

    U,S,V = scipy.linalg.svd(np.array(coefs))

    # normalized eigenvalues
    plt.figure(figsize=[4,3])
    plt.bar(np.arange(len(S))+1, S/np.sum(S))
    plt.ylim([0,1])
    plt.xlabel('SV')
    plt.ylabel('contribution')
    plt.show()

    # eigenvectors 
    plt.figure(figsize=[21,3])
    for i in range(len(S)):
        plt.subplot(1,len(coefs),i+1)
        plt.barh(-np.arange(len(params)),V[i,:])
        plt.yticks(-np.arange(len(params)), params)
        plt.title('SV'+str(i+1))
    plt.tight_layout()
    plt.show()

    # low dim projections
    fig = plt.figure(figsize=[12,4])
    # 2-d
    ax = fig.add_subplot(1,3,1)
    for i,(x,y) in enumerate((coefs@V)[:,:2]):
        ax.scatter(x,y, label=behavs[i])
    plt.xlabel('SV1')
    plt.ylabel('SV2')
    plt.legend(behavs)

    ax = fig.add_subplot(1,3,2)
    for i,(x,y) in enumerate((coefs@V)[:,1:3]):
        ax.scatter(x,y, label=behavs[i])
    plt.xlabel('SV2')
    plt.ylabel('SV3')
    plt.legend(behavs)
    
    ax = fig.add_subplot(1,3,3)
    for i,(x,y) in enumerate((coefs@V)[:,[0,2]]):
        ax.scatter(x,y, label=behavs[i])
    plt.xlabel('SV1')
    plt.ylabel('SV3')
    plt.legend(behavs)
    plt.tight_layout()
    plt.show()

    # 3-d
    fig = plt.figure(figsize=[8,4])
    ax = fig.add_subplot(projection='3d')
    for i,(x,y,z) in enumerate((coefs@V)[:,:3]):
        ax.scatter(x,y,z, label=behavs[i])
    plt.xlabel('SV1')
    plt.ylabel('SV2')
    ax.set_zlabel('SV3')
    plt.legend(behavs)
    plt.tight_layout()
    plt.show()


def create_df_null(multivar, multivar_null):
    """ create pandas dataframe of regression coefficients w.r.t. null distribution """
    behavs = multivar.keys()
    for behav in behavs:
        params = multivar[behav]['Ridge']['model'].feature_names_in_
        nulls = []
        for model_name in multivar_null[behav].keys():
            model = multivar_null[behav][model_name]['model']
            nulls.append(dict((f,v) for f,v in zip(model.feature_names_in_, model.coef_)))
        df_null = pd.DataFrame(nulls)
        df_null['null'] = True

        # create df of regression coefficients from full dataset (index=0) and all CV folds (index=1:)
        coefs = [dict((f,v) for f,v in zip(multivar[behav]['Ridge']['model'].feature_names_in_, multivar[behav]['Ridge']['model'].coef_))]
        for est in multivar[behav]['Ridge']['cv_results']['estimator']:
            coefs.append(dict((f,v) for f,v in zip(multivar[behav]['Ridge']['model'].feature_names_in_, est.coef_)))
        df_sim = pd.DataFrame(coefs)
        df_sim['null'] = False

        df = pd.concat([df_null, df_sim], ignore_index=True)

        multivar[behav]['Ridge']['df_null'] = df


def get_param_stats_title(df, params, behav):
    """ stats on null distribs with two samples """
    ttl = behav+"\n"
    for par in params:
        x = df[~df.null].iloc[1:][par]
        y = df[df.null][par]

        x_norm, y_norm = scipy.stats.normaltest(x), scipy.stats.normaltest(y)
        utest = scipy.stats.mannwhitneyu(x,y)
        U, p = utest.statistic, utest.pvalue
        ttl += "U={}, p={:.4f}                ".format(int(U), p)
        #print(par)
        #print('normal test: ', scipy.stats.normaltest(x), scipy.stats.normaltest(y))
        #print('t-test: ', scipy.stats.ttest_ind(x,y))
        #print('U-test: ', scipy.stats.mannwhitneyu(x,y))
        #print('Kruskal-Wallis: ', scipy.stats.kruskal(x,y))
    return ttl


def plot_null_distrib(multivar, args=None):
    """ plot null distributions of regression coefficients and stats """
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.size':11, 'axes.titlesize':'medium', 'mathtext.default': 'regular'})

    behavs = list(multivar.keys())
    params = multivar[behavs[0]]['Ridge']['model'].feature_names_in_
    
    ## COEFFICIENTS FROM FULL REGRESSION

    # first figure: individual axes for each params + stats
    for j,behav in enumerate(behavs):
        fig = plt.figure(figsize=[2*len(params),3])
        df = multivar[behav]['Ridge']['df_null']
        for i,par in enumerate(params):
            ax = fig.add_subplot(1, len(params), i+1)
            n, mu, sigma = len(df[df.null][par]), df[df.null][par].mean(), df[df.null][par].std()
            t = (float(df[~df.null].iloc[0][par]) - mu) / sigma
            p = scipy.stats.t.sf(np.abs(t), n-1)                                        
            sbn.swarmplot(df[df.null][par], ax=ax, color='black', alpha=0.1, size=1)
            sbn.swarmplot([df[~df.null].iloc[0][par]], ax=ax, color='red', alpha=1)
            plt.title("t={:.2f} p={:.3f}".format(t,p))
        plt.suptitle(behav)
        plt.tight_layout()
        if args.save_figs:
            fname = os.path.join(proj_dir, 'img', 'null_plots_indiv_stats_'+behav+today()+'.svg')
            plt.savefig(fname)
        if args.plot_figs:
            plt.show(block=False)
        else:
            plt.close()

    if args.plot_cv_null_distrib:
        # second figure: single axe for all params, no stats 
        fig = plt.figure(figsize=[15,5*len(behavs)])
        for j,behav in enumerate(behavs):
            df = multivar[behav]['Ridge']['df_null']
            ax = fig.add_subplot(len(behavs), 1, j+1)
            sbn.swarmplot(df.melt(id_vars='null'), x='variable', y='value', hue='null', dodge=True, palette=['black', 'gainsboro'], ax=ax)
            ax.get_legend().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ttl = get_param_stats_title(df, params, behav)
            plt.title(ttl)
        plt.tight_layout()
        if args.save_figs:
            fname = os.path.join(proj_dir, 'img', 'null_plots_all_behavs'+today()+'.svg')
            plt.savefig(fname)
        if args.plot_figs:
            plt.show(block=False)
        else:
            plt.close()
        plt.show(block=False)


def plot_behavs_distrib(df_sim_pat, behavs):
    """ Plot behavioral score distribution with median line"""
    plt.figure(figsize=[20,4])
    for i,behav in enumerate(behavs):
        plt.subplot(1,len(behavs),i+1)
        plt.hist(df_sim_pat[behav], bins=20, alpha=0.5)
        counts, bins = np.histogram(df_sim_pat[behav], bins=20)
        plt.vlines(np.median(df_sim_pat[behav]), min(counts), max(counts), color='black')
        plt.xlabel(behav)
    plt.tight_layout()
    plt.show()


def discretize_behavs(df_sim_pat, behavs):
    """ """
    discretize = lambda x: 0 if x else 1
    behavs_discrete = []
    for behav in behavs:
        new_behav = behav[:-5]+'_discrete'
        df_sim_pat[new_behav] = [discretize(v) for v in df_sim_pat[behav]<df_sim_pat[behav].median()]
        behavs_discrete.append(new_behav)
    return behavs_discrete


def print_ANOVA(df_sim_pat, behavs, params):
    """ Print stats for mixed and one-way ANOVAs """
    behavs_discrete = discretize_behavs(df_sim_pat, behavs)
    print("Mixed ANOVAs:")
    for behav in behavs_discrete:
        print("\n"+behav)
        print(df_sim_pat[np.concatenate([[behav, 'subj'], params])].melt(id_vars=[behav, 'subj']).mixed_anova(dv='value', within='variable', between=behav, subject='subj'))

    print("One-way ANOVAs:")
    for behav in behavs_discrete:
        print("\n"+behav)
        for param in params:
            print("\n"+param)
            print(pg.normality(dv=param, group=behav, data=df_sim_pat[np.concatenate([[behav, 'subj', param]])]))
            print(pg.homoscedasticity(dv=param, group=behav, data=df_sim_pat[np.concatenate([[behav, 'subj', param]])]))
            print('\nANOVA')
            print(df_sim_pat[np.concatenate([[behav, 'subj', param]])].anova(dv=param, between=behav, detailed=False))
            print('\nWelch ANOVA')
            print(df_sim_pat[np.concatenate([[behav, 'subj', param]])].welch_anova(dv=param, between=behav))
            print('\nKruskal')
            print(pg.kruskal(dv=param, between=behav, data=df_sim_pat[np.concatenate([[behav, 'subj', param]])]))

#--------------------------#
#   RESTORATION ANALYSIS   #
#--------------------------#

def get_df_base(args):
    """ Import simulations from infered parameters for controls and patients without restoration """
    if args.use_optim_params:
        fname = 'sim_base_optim_20240326.db'
    else:
        fname = 'sim_base_20240320.db'
    #with sl.connect(os.path.join(proj_dir, 'postprocessing', 'sim_test_20230614.db')) as conn:
    with sl.connect(os.path.join(proj_dir, 'postprocessing', fname)) as conn:
        #df = pd.read_sql("SELECT * FROM SIMTEST WHERE test_param='None'", conn)
        df = pd.read_sql("SELECT * FROM SIMTEST", conn)
        df.test_param = 'None'
    conn.close()
    return df

def get_df_other_base(args):
    """ Import simulations from infered parameters for controls and patients without restoration 
    (2nd batch to be able to compare cntrols to controls and patients to patients with same n) """
    if args.use_optim_params:
        fname = 'sim_base_optim_20240327.db'
    else:
        fname = 'sim_base_20240321.db'
    #with sl.connect(os.path.join(proj_dir, 'postprocessing', 'sim_test_20230614.db')) as conn:
    with sl.connect(os.path.join(proj_dir, 'postprocessing', fname)) as conn:
        #df = pd.read_sql("SELECT * FROM SIMTEST WHERE test_param='None'", conn)
        df = pd.read_sql("SELECT * FROM SIMTEST", conn)
        df.test_param = 'None'
    conn.close()
    return df

def get_df_3rd_base(args):
    """ Import simulations from infered parameters for controls and patients without restoration 
    (2nd batch to be able to compare cntrols to controls and patients to patients with same n) """
    fname = 'sim_base_optim_20240328.db'
    with sl.connect(os.path.join(proj_dir, 'postprocessing', fname)) as conn:
        df = pd.read_sql("SELECT * FROM SIMTEST", conn)
        df.test_param = 'None'
    conn.close()
    return df

def fix_df_base(df_base):
    """ fix duplicate entries in df_base """
    pathways = df_base.pathway.unique()
    def iter_df():
        inds = list(itertools.islice(range(len(df_base)), 0, None, len(pathways)))
        for i in inds:
            df_line = df_base.iloc[i].to_frame().transpose()
            yield df_base.iloc[i:i+len(pathways)].pivot(index='subj', columns='pathway', values='corr').reset_index().merge(df_line, on='subj')

    new_rows = [row for row in iter_df()]
    new_df = pd.concat(new_rows).reset_index().drop(columns=['index', 'level_0', 'pathway', 'corr'])
    new_df['subj'] = new_df.apply(lambda row: "sim-test{:06d}".format(int(row.name)+1), axis=1)
    return new_df

def get_restoration_suffix(args):
    suffix = '_'+args.distance_metric
    if args.use_optim_params:
        suffix += '_optim'
    suffix += '_'+args.efficacy_base
    return suffix


def compute_rmse_restore_data(df_data, df_sims, args):
    """ [DEPRECATED] Compute Root Mean Squared Error between the data and batches from the simulation """
    pathways = df_data.pathway.unique()
    n_folds = int(np.floor(len(df_sims) / args.n_sims))
    RMSEs = []
    for i in np.arange(0,n_folds*args.n_sims, args.n_sims):
        RMSE = []
        for pathway in pathways:
            mu_data, sigma_data = df_data[df_data.pathway==pathway]['corr'].mean(), df_data[df_data.pathway==pathway]['corr'].std()
            mu_sims, sigma_sims = df_sims.iloc[i:i+args.n_sims][pathway].mean(), df_sims.iloc[i:i+args.n_sims][pathway].std()
            RMSE.append((mu_data-mu_sims)**2 + (sigma_data-sigma_sims)**2)
        RMSEs.append(np.sqrt(np.sum(RMSE)))
    return np.array(RMSEs)


def compute_distance_restore_sims(df_base, df_sims, args):
    """ Compute distance metric between batches of simulations in controls, patients and 
    continuous restoration from patients to controls.
    
    Parameters
    ----------
        df_base: pandas.DataFrame
            Simulated data from virtual OCD subjects (patients) and healthy controls. 
            No virtual interventions were performed. Corresponds to simulation using OCD and healthy posteriors. 
        df_sims: pandas.DataFrame
            Simulation outputs loaded from the virtual interventions database.
        args: argparse.Namespace
            Extra arguments with options.

    Returns
    -------
        outputs: list of dictionanries
            Distances between simulated interventions and healthy controls cohorts, 
            with normalized parameters values w.r.t original OCD parameter posterior distribution.

    """
    max_len = np.min([len(df_sims), len(df_base[df_base.base_cohort=='controls'])])
    n_folds = int(np.floor(max_len/ args.n_sims))
    i_s = np.arange(0,n_folds*args.n_sims, args.n_sims)
    outputs = []
    indices = itertools.product(i_s, i_s)
    df_other_base = get_df_other_base(args)
    if args.use_optim_params:
        indices = zip(i_s, i_s)
    for i,j in indices:
        # get FC distance from controls or patients FC  
        sim = df_sims.iloc[i:i+args.n_sims]
        base_cons = df_base[df_base.base_cohort=='controls'].iloc[j:j+args.n_sims]
        base_pats = df_base[df_base.base_cohort=='patients'].iloc[j:j+args.n_sims]
        base_ocons = df_other_base[df_other_base.base_cohort=='controls'].iloc[i:i+args.n_sims]
        base_opats = df_other_base[df_other_base.base_cohort=='patients'].iloc[i:i+args.n_sims]

        #distance = metric[args.distance_metric](sim[args.pathways], base_cons[args.pathways])
        distance_post_pre = metric[args.distance_metric](sim[args.pathways], base_pats[args.pathways])
        distance_pre_hc = metric[args.distance_metric](base_cons[args.pathways], base_pats[args.pathways])
        distance_post_hc = metric[args.distance_metric](sim[args.pathways], base_cons[args.pathways])
        distance_hc_hc = metric[args.distance_metric](base_ocons[args.pathways], base_cons[args.pathways])
        distance_pre_pre = metric[args.distance_metric](base_opats[args.pathways], base_pats[args.pathways])

        # get parameter difference from patients parameters
        output = dict(('z_'+param, ztest(x1=sim[param], x2=base_pats[param])[0]) 
                      for param in args.params)
        
        
        output['dist'] = distance_post_hc
        output['dist_pre_hc'] = distance_pre_hc
        output['dist_post_pre'] = distance_post_pre
        output['dist_pre_pre'] = distance_pre_pre
        output['dist_hc_hc'] = distance_hc_hc
        #output['efficacy'] = distance_post_pre/distance_pre_hc

        if args.use_optim_params:
            euclidian_post_hc = paired_euclidian(sim[args.pathways], base_cons[args.pathways])
            euclidian_hc_hc = paired_euclidian(base_ocons[args.pathways], base_cons[args.pathways])
            euclidian_pre_hc = paired_euclidian(base_pats[args.pathways], base_cons[args.pathways])
            #output['paired_tstat_T'], output['paired_tstat_pval'] = scipy.stats.ttest_rel(euclidian_post_hc, euclidian_hc_hc)
            #output['paired_wilcoxon'], output['paired_wilcoxon_pval'] = scipy.stats.wilcoxon(euclidian_post_hc, euclidian_hc_hc)
            output['paired_tstat_T'], output['paired_tstat_pval'] = scipy.stats.ttest_rel(euclidian_pre_hc, euclidian_post_hc)
            output['paired_wilcoxon'], output['paired_wilcoxon_pval'] = scipy.stats.wilcoxon(euclidian_pre_hc, euclidian_post_hc)
        outputs.append(output)
    return outputs


def compute_distance_restore(df_sims, args):
    """ Compare base simulations (from OCD subjects and healthy controls) to simulated interventions by
    calculating distances in parallel. 
    
    Parameters
    ----------
        df_sims: pandas.DataFrame
            Simulation outputs loaded from the database.
        args: argparse.Namespace
            Extra arguments with options.
    
    Returns
    -------
        df_restore: pandas.DataFrame
            Copy of df_sims with efficacy measures (including cohorts first level statistics).
    
    """
    # Test params are the parameters to be permuted.
    test_params = df_sims[df_sims.n_test_params<=args.n_test_params].test_param.unique()
    
    # df_base are the simulated using OCD subjects and helthy controls' posterior (no invervention).
    df_base = get_df_base(args)
    #df_base = fix_df_base(df_base)

    distance_outputs = Parallel(n_jobs=args.n_jobs, backend='loky',verbose=3)(delayed(compute_distance_restore_sims)
                                        (df_base=df_base, 
                                        df_sims=df_sims[(df_sims.base_cohort=='patients') 
                                                         & (df_sims.test_cohort=='controls') 
                                                         & (df_sims.test_param==test_param)], 
                                         args=args) 
                                        for test_param in test_params)

    
    # Add first order statistics of the cohort's distance to base FC
    lines = []
    for tp,outputs in zip(test_params, distance_outputs):
        df_test_param = pd.DataFrame(outputs)
        df_test_param['test_param'] = tp
        df_test_param['mean'] = np.mean(df_test_param.dist)
        df_test_param['median'] = np.median(df_test_param.dist)
        df_test_param['std'] = np.std(df_test_param.dist)
        df_test_param['tstat'], df_test_param['pval'] = scipy.stats.ttest_ind(df_test_param.dist_pre_hc, df_test_param.dist)
        df_test_param['ustat'], df_test_param['upval'] = scipy.stats.mannwhitneyu(  np.array(df_test_param.dist_pre_hc), 
                                                                                    np.array(df_test_param.dist),
                                                                                    alternative='greater')
        
        # non-paramteric effect size (https://aakinshin.net/posts/nonparametric-effect-size/)
        Q_x = np.median(np.array(df_test_param.dist_pre_hc))
        n_x = np.array(df_test_param.dist_pre_hc).shape[0]
        Q_y = np.median(np.array(df_test_param.dist))
        n_y = np.array(df_test_param.dist).shape[0]
        MAD_x = scipy.stats.median_abs_deviation(np.array(df_test_param.dist_pre_hc))
        MAD_y = scipy.stats.median_abs_deviation(np.array(df_test_param.dist))
        PMAD_xy = np.sqrt(((n_x-1)*MAD_x**2 + (n_y-1)*MAD_y**2) / (n_x + n_y -2))
        df_test_param['gamma_es'] = (Q_x - Q_y) / PMAD_xy

        lines.append(df_test_param)
        
    df_restore = pd.concat(lines, ignore_index=True)
    # add a column with number of targeted parameters
    df_restore['n_test_params'] = df_restore.test_param.apply(lambda pars_str: len(pars_str.split(' ')) if pars_str!='None' else 0)
    return df_restore
 

def get_max_distance_data(args):
    """ Compute the distance between controls and patients in real data """
    df_fc_pre_post = load_dist_to_FC_controls(args)
    df_fc_pre = df_fc_pre_post[df_fc_pre_post.ses=='ses-pre'] 
    #pathways = np.sort(df_data.pathway.unique())
    #df_tmp = df_data.pivot(index=['subj', 'cohort'], columns='pathway', values='corr').reset_index()
    cons,pats = df_fc_pre[df_fc_pre.cohort=='controls'],df_fc_pre[df_fc_pre.cohort=='patients']
     
     # compute euclidian distances to mean controls
    con_ref = cons[args.pathways].apply(np.mean, axis=0)
    pat_ref = pats[args.pathways].apply(np.mean, axis=0)
    def dist_con(x):
        return np.sqrt(np.sum((np.array(x)-np.array(con_ref))**2))
    def dist_pat(x):
        return np.sqrt(np.sum((np.array(x)-np.array(pat_ref))**2))
    
    distances = {'con':[], 'pat':[], 'con_pat':[]}
    distances['con'] = cons[args.pathways].apply(dist_con, axis=1)
    distances['pat'] = pats[args.pathways].apply(dist_pat, axis=1)
    distances['con_pat'] = pats[args.pathways].apply(dist_con, axis=1)
    return distances


def get_max_distance_sims(args):
    """ Compute the distance between controls and patients in simulated dataset.
        This function is also used to compute the null distributions.    
    """
    df_base = get_df_base(args)

    # 2nd base simulation to compute the patients' null 
    df_other_base = get_df_other_base(args)
    #df_base = fix_df_base(df_base)
    #n_pathways = len(df.pathway.unique())

    # 3rd base simulation to compute the controls' null
    df_3rd_base = get_df_3rd_base(args)

    cons = df_base[df_base.base_cohort=='controls']
    pats = df_base[df_base.base_cohort=='patients']

    o_cons = df_other_base[df_other_base.base_cohort=='controls']
    o_pats = df_other_base[df_other_base.base_cohort=='patients']

    oo_cons = df_3rd_base[df_3rd_base.base_cohort=='controls']
    oo_pats = df_3rd_base[df_3rd_base.base_cohort=='patients']

    distances = {'con':[], 'pat':[], 'con_pat':[], 'con_pat_centroid':0, 'con_con':[], 'pat_pat':[], 'pat_con':[], 'con_ocon':[], \
                 'conpat':[],'conopat':[], 'oconpat':[], 'conocon':[], 'conoocon':[], 'opatpat':[], 
                 'tstat':0, 'pval':0, 'tstat_':0, 'pval_':0,
                 'ustat':0, 'upval':0, 'ustat_':0, 'upval_':0, 'gamma_es':0, 'gamma_es_':0, 
                 'paired_tstat_T_concon':[], 'paired_tstat_pval_concon':[],
                 'paired_tstat_T_patpat':[], 'paired_tstat_pval_patpat':[],
                 'paired_tstat_T_conpat':[], 'paired_tstat_pval_conpat':[]}
    i_s = list(itertools.islice(range(len(cons)), 0, None, args.n_sims))
    
    # within-group distances (using same base)
    for i,j in itertools.combinations(i_s, 2):
        cons_i = cons.iloc[i:i+args.n_sims]
        cons_j = cons.iloc[j:j+args.n_sims]
        pats_i = pats.iloc[i:i+args.n_sims]
        pats_j = pats.iloc[j:j+args.n_sims]
        distances['con'].append(metric[args.distance_metric](cons_i[args.pathways], cons_j[args.pathways]))
        distances['pat'].append(metric[args.distance_metric](pats_i[args.pathways], pats_j[args.pathways]))
    
    # between-group distances
    for i,j in itertools.product(i_s, i_s):
        pats_i = pats.iloc[i:i+args.n_sims]
        cons_j = cons.iloc[j:j+args.n_sims]
        distances['con_pat'].append(metric[args.distance_metric](pats_i[args.pathways], cons_j[args.pathways]))
        pats_i = o_pats.iloc[i:i+args.n_sims]
        distances['pat_con'].append(metric[args.distance_metric](pats_i[args.pathways], cons_j[args.pathways]))
    distances['con_pat_centroid'] = metric[args.distance_metric](cons[args.pathways], pats[args.pathways])

    # within-group distances (using other base)
    for i,j in itertools.product(i_s, i_s):
        cons_i = cons.iloc[i:i+args.n_sims]
        cons_j = o_cons.iloc[j:j+args.n_sims]
        distances['con_con'].append(metric[args.distance_metric](cons_i[args.pathways], cons_j[args.pathways]))
        cons_i = oo_cons.iloc[i:i+args.n_sims]
        distances['con_ocon'].append(metric[args.distance_metric](cons_i[args.pathways], cons_j[args.pathways]))
        
        pats_i = pats.iloc[i:i+args.n_sims]
        pats_j = o_pats.iloc[j:j+args.n_sims]
        distances['pat_pat'].append(metric[args.distance_metric](pats_i[args.pathways], pats_j[args.pathways]))

    distances['tstat'], distances['pval'] = scipy.stats.ttest_ind(np.array(distances['con_pat']), np.array(distances['con_con']))
    distances['tstat_'], distances['pval_'] = scipy.stats.ttest_ind(np.array(distances['con_pat']), np.array(distances['pat_con']))
    distances['ustat'], distances['upval'] = scipy.stats.mannwhitneyu(np.array(distances['con_pat']), np.array(distances['con_con']), alternative='greater')
    distances['ustat_'], distances['upval_'] = scipy.stats.mannwhitneyu(np.array(distances['con_pat']), np.array(distances['pat_con']), alternative='greater')

    # non-parametric effect size
    Q_x = np.median(np.array(distances['con_pat']))
    n_x = np.array(np.array(distances['con_pat'])).shape[0]
    Q_y = np.median(np.array(distances['con_con']))
    n_y = np.array(distances['con_con']).shape[0]
    MAD_x = scipy.stats.median_abs_deviation(np.array(distances['con_pat']))
    MAD_y = scipy.stats.median_abs_deviation(np.array(distances['con_con']))
    PMAD_xy = np.sqrt(((n_x-1)*MAD_x**2 + (n_y-1)*MAD_y**2) / (n_x + n_y -2))
    distances['gamma_es'] = (Q_y - Q_x) / PMAD_xy

    Q_y = np.median(np.array(distances['pat_con']))
    n_y = np.array(distances['pat_con']).shape[0]
    MAD_y = scipy.stats.median_abs_deviation(np.array(distances['pat_con']))
    PMAD_xy = np.sqrt(((n_x-1)*MAD_x**2 + (n_y-1)*MAD_y**2) / (n_x + n_y -2))
    distances['gamma_es_'] = (Q_y - Q_x) / PMAD_xy

    # paired distances (optim)
    for i,j in zip(i_s, i_s):
        cons_ = cons.iloc[i:i+args.n_sims]
        pats_ = pats.iloc[i:i+args.n_sims]
        ocons_ = o_cons.iloc[j:j+args.n_sims]
        opats_ = o_pats.iloc[j:j+args.n_sims]
        oocons_ = oo_cons.iloc[j:j+args.n_sims]
        oopats_ = oo_pats.iloc[j:j+args.n_sims]
        distances['conpat'].append(metric[args.distance_metric](pats_[args.pathways], cons_[args.pathways]))
        distances['conopat'].append(metric[args.distance_metric](opats_[args.pathways], cons_[args.pathways]))
        distances['oconpat'].append(metric[args.distance_metric](pats_[args.pathways], ocons_[args.pathways]))
        distances['conocon'].append(metric[args.distance_metric](ocons_[args.pathways], cons_[args.pathways]))
        distances['conoocon'].append(metric[args.distance_metric](oocons_[args.pathways], cons_[args.pathways]))
        distances['opatpat'].append(metric[args.distance_metric](pats_[args.pathways], opats_[args.pathways]))

        d_conocon = paired_euclidian(ocons_[args.pathways], cons_[args.pathways])
        d_conoocon = paired_euclidian(oocons_[args.pathways], cons_[args.pathways])
        T, p = scipy.stats.ttest_rel(d_conocon, d_conoocon)
        distances['paired_tstat_T_concon'].append(T) 
        distances['paired_tstat_pval_concon'].append(p)
        d_patopat = paired_euclidian(opats_[args.pathways], pats_[args.pathways])
        d_patoopat = paired_euclidian(oopats_[args.pathways], pats_[args.pathways])
        T, p = scipy.stats.ttest_rel(d_patopat, d_patoopat)
        distances['paired_tstat_T_patpat'].append(T) 
        distances['paired_tstat_pval_patpat'].append(p)
        d_conpat = paired_euclidian(pats_[args.pathways], cons_[args.pathways], )
        T, p = scipy.stats.ttest_rel(d_conpat, d_patopat)
        distances['paired_tstat_T_conpat'].append(T) 
        distances['paired_tstat_pval_conpat'].append(p) 
    
    return distances


def plot_efficacy_transform(args):
    """ Plot distance and efficacy distrib and KDEs"""
    
    distances = get_max_distance_sims(args)

    def get_KDE(values):
        vmin, vmax = np.min(values), np.max(values)
        bw = (vmax-vmin)/20
        kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(values).reshape(-1,1))
        X = np.linspace(vmin-3*bw, vmax+3*bw, 100).reshape(-1,1)
        pdf = kde.score_samples(X)
        return {'kde':kde, 'pdf':pdf, 'X':X}
    
    #kde_con = get_KDE(distances['con'])
    #kde_pat = get_KDE(distances['pat'])
    #bias_con = np.mean(distances['con'])
    #bias_pat = np.mean(distances['pat'])
    bias_con = np.mean(distances['con'])
    bias_pat = np.mean(distances['pat'])
    bias_con_pat = np.mean(distances['con_pat'])
    bias = np.mean([bias_con, bias_pat])
    offset = np.mean(distances['con_pat'])
    #scale = np.mean(distances['con_pat'])
    scale = np.mean(distances['con_pat'])

    kde_con = get_KDE(1-((distances['con']-bias_con)/scale)*100)
    kde_pat = get_KDE(1-((distances['con_pat']-bias_con)/scale)*100)

    plt.figure(figsize=[10,12])
    ax = plt.subplot(5,1,1)
    #plt.hist(distances['pat']-np.mean(distances['pat'])+offset, color='orange', alpha=0.5, density=False, bins=np.linspace(0,0.3,30))
    #plt.hist(distances['con']-np.mean(distances['con']), bins=np.linspace(0,0.3,30), color='lightblue', alpha=0.5, density=False)
    plt.hist(distances['con'], bins=np.linspace(0,0.3,30), color='lightblue', alpha=0.3, density=False)
    plt.hist(distances['pat'], bins=np.linspace(0,0.3,30), color='orange', alpha=0.3, density=False)
    plt.hist(distances['con_pat']-bias_con-bias_pat, bins=np.linspace(0,0.3,30), color='magenta', alpha=0.3, density=False)
    plt.vlines(np.mean(distances['con']), ymin=0, ymax=50, color='lightblue')
    plt.vlines(np.mean(distances['pat']), ymin=0, ymax=50, color='orange')
    plt.vlines(np.mean(distances['con_pat']), ymin=0, ymax=50, color='magenta')
    #plt.xlim([0,1])
    plt.xlabel('$d$', fontsize=12)
    plt.ylabel('counts', fontsize=12)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    
    ax = plt.subplot(5,1,2)
    plt.hist(distances['con']-bias_con, bins=np.linspace(-0.1,0.3,40), color='lightblue', alpha=0.5, density=False)
    plt.hist(distances['con_pat'], bins=np.linspace(-0.1,0.3,40), color='orange', alpha=0.5, density=False)
    #plt.xlim([0,1])
    plt.xlabel('$d-\mu_{d_{XX}}$', fontsize=12)
    plt.ylabel('counts', fontsize=12)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    
    ax = plt.subplot(5,1,3)
    plt.hist((distances['con']-bias_con)/scale, bins=np.linspace(-2,2,40), color='lightblue', alpha=0.5, density=False)
    plt.hist((distances['con_pat']-bias_con)/scale, bins=np.linspace(-2,2,40), color='orange', alpha=0.5, density=False)
    #plt.xlim([0,1])
    plt.xlabel('$ \\frac{d-\mu_{d_{XX}}}{\mu_{d_{XY}} - \mu_{d_{XX}}}$', fontsize=12)
    plt.ylabel('counts', fontsize=12)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    ax = plt.subplot(5,1,4)
    plt.hist(1-((distances['con']-bias_con)/scale), bins=np.linspace(-2,2,40), color='lightblue', alpha=0.5, density=False)
    plt.hist(1-((distances['con_pat']-bias_con)/scale), bins=np.linspace(-2,2,40), color='orange', alpha=0.5, density=False)
    #plt.xlim([0,1])
    plt.xlabel('$1- \\frac{d-\mu_{d_{XX}}}{\mu_{d_{XY}} - \mu_{d_{XX}}}$', fontsize=12)
    plt.ylabel('counts', fontsize=12)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    
    ax = plt.subplot(5,1,5)
    #plt.plot(100*(kde_pat['X']-bias)/scale, kde_pat['pdf'], color='orange')
    #plt.plot(100*(kde_con['X']-bias+offset)/scale, kde_con['pdf'], color='lightblue')
    plt.plot(kde_con['X'], kde_con['pdf'], color='lightblue')
    plt.plot(kde_pat['X'], kde_pat['pdf'], color='orange')
    plt.xlabel('$E_{ff} \, (\%)$', fontsize=12)
    plt.ylabel('density', fontsize=12)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    plt.tight_layout()

    if args.save_figs:
        fname = 'efficacy_trasnform'+today()+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))
        fname = 'efficacy_trasnform'+today()+'.png'
        plt.savefig(os.path.join(proj_dir, 'img', fname))
    plt.show()


def format_param(param):
    """ return LaTeX formated string of parameter (without dollar signs) """
    param = param_mapping[param]
    par_ = param.split('_')

    # handle greek letters
    if (param.startswith('sigma') or param.startswith('eta')):
        par_[0] = r"\\"[0]+par_[0]
    
    # handle C_
    if len(par_)==2:
        par_[1] = r'{'+par_[1]+r'}'
    
    elif len(par_)==3:
        par_[1] = r'{'+par_[1]
        par_[2] = r'{'+par_[2]+r'}}'

    formatted_param = '_'.join(par_)
    return formatted_param


def format_labels(labels):
    new_labels = []
    for label in labels:
        new_label = []
        params = label.get_text().split(' ')
        for param in params:
            formatted_param = format_param(param)
            new_label.append(formatted_param)
        new_label = '${}$'.format(r'\quad'.join(new_label))
        #label.set_text(new_label)
        new_labels.append(new_label)
    return new_labels


def compute_efficacy(df_restore, args=None):
    """ Add new column to input DataFrame with treatment efficacy based on distance to healthy controls
    in functional connectivity space. 
        
    .. note::
        Simulation metric is the Wasserstein distance and the data metric is the Euclidean distance. 
        Comparision in FC space is automatically performed using the correct metric (default: Wasserstein). 

    Parameters
    ----------
        df_restore: pandas.DataFrame
            Virtual intervention simulation outputs with distance precomputed. 
        args: argparse.Namespace
            Extra arguments with options. Important option in this function is `args.efficacy_base` which informs 
            how treatment efficacy is computed (e.g. retained was "ustat")

    Returns
    -------
        df_restore: pandas.DataFrame
            Input DataFrame with new column 'efficacy'.

    """
    if args.efficacy_base=='sims':
        distances = get_max_distance_sims(args)
        # mu_d
        offset = np.mean([np.mean(distances['pat']), np.mean(distances['con'])])
        # mu_d_XY
        #scale = np.mean(distances['con_pat'])
        scale = distances['con_pat_centroid']
        # E_ff
        #df_restore['efficacy'] = (1-((df_restore['dist']-offset))/scale)*100 # make % of restoration
        df_restore['efficacy'] = (1-df_restore['dist']/scale)*100 # make % of restoration
    
    elif args.efficacy_base=='paired':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                df_restore.loc[ids, 'efficacy'] = (1-np.divide(df_restore.loc[ids].dist, distances['con_pat']))*100
    
    elif args.efficacy_base=='paired_A':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                df_restore.loc[ids, 'efficacy'] = np.divide(df_restore.loc[ids].dist, distances['con_pat'])

    elif args.efficacy_base=='paired_B':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                df_restore.loc[ids, 'efficacy'] = np.divide(distances['con_pat'],df_restore.loc[ids].dist)

    elif args.efficacy_base=='paired_C':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                if args.use_optim_params:
                    df_restore.loc[ids, 'efficacy'] = 1-np.divide(df_restore.loc[ids].dist, distances['conpat'])
                else:
                    df_restore.loc[ids, 'efficacy'] = 1-np.divide(df_restore.loc[ids].dist, distances['con_pat'])

    elif args.efficacy_base=='paired_D':
        distances = get_max_distance_sims(args)
        # transpose indices to matched "pat" (pre) pairs instead of "con" (hc)
        n = int(np.sqrt(len(distances['con_pat'])))
        con_pat = np.array(distances['con_pat']).reshape((n,n)).T.ravel()
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                df_restore.loc[ids, 'efficacy'] = np.divide(df_restore.loc[ids].dist_post_pre, con_pat)

    elif args.efficacy_base=='paired_E':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                df_restore.loc[ids, 'efficacy'] = df_restore.loc[ids].dist

    elif args.efficacy_base=='paired_F':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                if args.use_optim_params:
                    df_restore.loc[ids, 'efficacy'] = np.divide(distances['conpat'] - df_restore.loc[ids].dist, distances['conpat'])
                else:
                    df_restore.loc[ids, 'efficacy'] = np.divide(distances['con_pat'] - df_restore.loc[ids].dist, distances['con_pat'])

    elif args.efficacy_base=='paired_G':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                if args.use_optim_params:
                    df_restore.loc[ids, 'efficacy'] = np.divide(np.mean(distances['conpat']) - df_restore.loc[ids].dist, np.mean(distances['conpat']))
                else:
                    df_restore.loc[ids, 'efficacy'] = np.divide(distances['con_pat_centroid'] - df_restore.loc[ids].dist, distances['con_pat_centroid'])

    # using T stat
    elif args.efficacy_base=='paired_H':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                if args.use_optim_params:
                    df_restore.loc[ids, 'efficacy'] = df_restore.loc[ids, 'paired_tstat_T']

    elif args.efficacy_base=='tstat':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                df_restore.loc[ids, 'efficacy'] = df_restore.loc[ids, 'tstat']

    elif args.efficacy_base=='ustat':
        distances = get_max_distance_sims(args)
        for test_param in df_restore.test_param.unique():
                ids = df_restore[df_restore.test_param==test_param].index
                df_restore.loc[ids, 'efficacy'] = np.array(df_restore.loc[ids, 'ustat'])/(400*400) # AUC1

    else:
        distances = get_max_distance_data(args)
        offset = np.mean([np.mean(distances['pat']), np.mean(distances['con'])])
        scale = np.mean(distances['con_pat'])
        df_restore['efficacy'] = (1-((df_restore['dist']))/scale)*100 # make % of restoration
    
    # add a column with number of targeted parameters
    df_restore['n_test_params'] = df_restore.test_param.apply(lambda pars_str: len(pars_str.split(' ')) if pars_str!='None' else 0)
    return df_restore


def get_df_top(sub_df_restore, args):
    # get top parameters that restore FC
    top_params = dict()
    top_params['all'] = sub_df_restore.sort_values('mean').test_param.unique()[:args.n_restore]
    if args.efficacy_base in ['paired_D', 'paired_E']:
        top_params['by_n'] = [sub_df_restore[sub_df_restore.n_test_params==n].sort_values('median').test_param.unique()[:args.n_tops]
                              for n in np.arange(1,args.n_test_params+1)]
    else:
        top_params['by_n'] = []
        for n in np.arange(1,args.n_test_params+1):
            for test_param in sub_df_restore[sub_df_restore.n_test_params==n].test_param.unique():
                inds = sub_df_restore[sub_df_restore.test_param==test_param].index
                sub_df_restore.loc[inds, 'median_Eff'] = np.median(sub_df_restore.iloc[inds]['efficacy'])
            top_params['by_n'].append(sub_df_restore[sub_df_restore.n_test_params==n].sort_values('median_Eff', ascending=False).test_param.unique()[:args.n_tops])
    top_params['by_n'] = np.concatenate(top_params['by_n'])
    df_top = sub_df_restore[sub_df_restore.test_param.apply(lambda x: x in top_params[args.sort_style])]   
    return df_top, top_params


def plot_distance_restore(df_restore, args, gs=None):
    """ Plot horizontal box plot of best virtual interventions outcomes, sorted by number of target points.
    
    Parameters
    ----------
        df_restore: pandas.DataFrame
            Virtual intervention simulation outputs with distance and efficacy precomputed. 
        args: (argparse.Namespace)
            Extra arguments with options. An important options here is args.n_tops which defines how many of the best 
            interventions to display by number of target points.
        gs: matplotlib.GridSpec
            (optional) A GridSpec object that can be used to embbed axes when this figure is a subplot 
            of a larger figure. 

    """
    # adds n_test_params column to dataframe 
    sub_df_restore = df_restore[df_restore.n_test_params<=args.n_test_params]

    # get top parameters that restore FC
    df_top, top_params = get_df_top(df_restore, args)
    
    # normalize distances to get efficacy 
    distances = get_max_distance_sims(args)
    
    # Patients "NULLS"
    df_pat = pd.DataFrame({'dist': distances['con_pat'], 'test_param':'patients', 'n_test_params': 0}) 
    n = int(np.sqrt(len(distances['con_pat'])))
    inds, = np.where(np.tril(np.ones((n,n)), k=-1).ravel())
    if args.efficacy_base=='paired_C':
        if args.use_optim_params:
            df_pat = pd.DataFrame({'dist': distances['conpat'], 'test_param':'patients', 'n_test_params': 0}) 
            df_pat['efficacy'] = 1 - np.divide(np.array(distances['conopat']), np.array(distances['conpat']))
        else:
            # get the con_pat matrice shifted so different patient to same control 
            con_pat = np.roll(np.array(distances['con_pat']).reshape(n,n), 1, axis=0).ravel()
            df_pat['efficacy'] = np.divide(df_pat['dist'] - con_pat, df_pat['dist'])
    elif args.efficacy_base=='paired_D':
        df_pat = pd.DataFrame({'dist': distances['pat'], 'test_param':'patients', 'n_test_params': 0}) 
        df_pat['efficacy'] = np.divide(df_pat['dist'], np.array(distances['con_pat'])[inds])
    elif args.efficacy_base=='paired_E':
        df_pat = pd.DataFrame({'dist': distances['con_pat'], 'test_param':'patients', 'n_test_params': 0}) 
        df_pat['efficacy'] = df_pat['dist']
    elif args.efficacy_base=='paired_F':
        if args.use_optim_params:
            df_pat = pd.DataFrame({'dist': distances['conpat'], 'test_param':'patients', 'n_test_params': 0}) 
            df_pat['efficacy'] = np.divide(np.array(distances['conpat']) - np.array(distances['conopat']), np.array(distances['conpat']))
        else:
            # get the con_pat matrice shifted so different patient to same control 
            con_pat = np.roll(np.array(distances['con_pat']).reshape(n,n), 1, axis=0).ravel()
            #df_pat['efficacy'] = np.divide(df_pat['dist'] - con_pat, df_pat['dist'])
            df_pat['efficacy'] = np.divide(np.array(distances['con_pat']) - np.array(distances['pat_con']), np.array(distances['con_pat']))
    elif args.efficacy_base=='paired_G':
        if args.use_optim_params:
            df_pat = pd.DataFrame({'dist': distances['conpat'], 'test_param':'patients', 'n_test_params': 0}) 
            df_pat['efficacy'] = np.divide(np.mean(distances['conpat']) - distances['conpat'], np.mean(distances['conpat']))
        else:
            df_pat = pd.DataFrame({'dist': distances['con_pat'], 'test_param':'patients', 'n_test_params': 0}) 
            # get the con_pat matrice shifted so different patient to same control 
            df_pat['efficacy'] = np.divide(distances['con_pat_centroid'] - df_pat['dist'], distances['con_pat_centroid'])
    
    elif args.efficacy_base=='paired_H':
        if args.use_optim_params:
            df_pat = pd.DataFrame({'dist': distances['conpat'], 'test_param':'patients', 'n_test_params': 0}) 
            df_pat['efficacy'] = distances['paired_tstat_T_patpat']
    
    elif args.efficacy_base=='tstat':
            df_pat = pd.DataFrame({'dist': np.unique(distances['tstat_']), 'test_param':'patients', 'n_test_params': 0}) 
            df_pat['efficacy'] = np.unique(distances['tstat_'])
    
    elif args.efficacy_base=='ustat':
            df_pat = pd.DataFrame({'dist': np.unique(distances['ustat_']), 'test_param':'patients', 'n_test_params': 0}) 
            df_pat['efficacy'] = np.unique(distances['ustat_']/(400*400)) # AUC
    
    else:
        df_pat['test_param'] = 'None'
        df_pat = compute_efficacy(df_pat, args=args)
        df_pat['test_param'] = 'patients'

    # Controls' NULLS
    df_con = pd.DataFrame({'dist': distances['con'], 'test_param':'controls', 'n_test_params': 0}) # test_param=controls for legend label
    n = int(np.sqrt(len(distances['con_pat'])))
    inds, = np.where(np.triu(np.ones((n,n)), k=1).ravel())
    if args.efficacy_base=='paired_A':
        df_con['efficacy'] = np.divide(df_con['dist'], np.array(distances['con_pat'])[inds])
    elif args.efficacy_base=='paired_B':
        df_con['efficacy'] = np.divide(np.array(distances['con_pat'])[inds], df_con['dist'])
    elif args.efficacy_base=='paired_C':
        if args.use_optim_params:
            df_con = pd.DataFrame({'dist': distances['conpat'], 'test_param':'controls', 'n_test_params': 0})
            df_con['efficacy'] = 1-np.divide(distances['conocon'], np.array(distances['conpat']))
        else:
            df_con['efficacy'] = 1-np.divide(df_con['dist'], np.array(distances['con_pat'])[inds])
    elif args.efficacy_base=='paired_D':
        df_con = pd.DataFrame({'dist': distances['con_pat'], 'test_param':'controls', 'n_test_params': 0}) 
        df_con['efficacy'] = np.divide(np.roll(np.array(distances['con_pat']).reshape(n,n),1).ravel(), df_con['dist'])
    elif args.efficacy_base=='paired_E':
        df_con = pd.DataFrame({'dist': distances['conocon'], 'test_param':'controls', 'n_test_params': 0}) 
        df_con['efficacy'] = distances['conocon']
    elif args.efficacy_base=='paired_F':
        if args.use_optim_params:
            df_con = pd.DataFrame({'dist': distances['conocon'], 'test_param':'controls', 'n_test_params': 0})
            #df_con['efficacy'] = 1 - np.array(distances['conocon']) #np.divide(np.array(distances['conpat']) - np.array(distances['conopat']), np.array(distances['conpat'])) 
            df_con['efficacy'] = 1 - np.divide(np.array(distances['conpat']) - np.array(distances['oconpat']), np.array(distances['conpat'])) 
        else:
            df_con = pd.DataFrame({'dist': distances['con_con'], 'test_param':'controls', 'n_test_params': 0})
            df_con['efficacy'] = 1 - np.divide(np.array(distances['con_pat']) - np.array(distances['con_con']), np.array(distances['con_pat']) )
    elif args.efficacy_base=='paired_G':
        if args.use_optim_params:
            df_con = pd.DataFrame({'dist': distances['conpat'], 'test_param':'controls', 'n_test_params': 0}) 
            df_con['efficacy'] = np.divide(np.mean(distances['conocon']) - distances['conocon'], np.mean(distances['conocon']))
        else:
            # get the con matrice shifted so different control to same control
            df_con['efficacy'] = np.divide(distances['con_pat_centroid'] - df_con['dist'], distances['con_pat_centroid'])
    
    elif args.efficacy_base=='paired_H':
        if args.use_optim_params:
            df_con = pd.DataFrame({'dist': distances['conocon'], 'test_param':'controls', 'n_test_params': 0}) 
            df_con['efficacy'] = distances['paired_tstat_T_conpat']

    elif args.efficacy_base=='tstat':
        df_con = pd.DataFrame({'dist': np.unique(distances['tstat']), 'test_param':'controls', 'n_test_params': 0}) 
        df_con['efficacy'] = np.unique(distances['tstat'])

        # check for normality
        for test_param in df_top.test_param.unique():
            df_ = df_top[df_top.test_param==test_param]
            stat,p = scipy.stats.normaltest(np.array(df_.dist))
            if p<0.05:
                print(test_param+": d(post,hc) is not normally distributed.")

    elif args.efficacy_base=='ustat':
        df_con = pd.DataFrame({'dist': np.unique(distances['ustat']), 'test_param':'controls', 'n_test_params': 0}) 
        df_con['efficacy'] = np.unique(distances['ustat']/(400*400)) # AUC
    

    # merge base and restore dataframes
    df_top = pd.concat([df_pat, df_con, df_top], ignore_index=True)
    top_params[args.sort_style] = np.concatenate([['patients','controls'],top_params[args.sort_style]])

    # plotting
    palette = {0: 'white', 1: 'lightpink', 2: 'plum', 3: 'mediumpurple', 4: 'lightsteelblue', 5:'skyblue', 6:'royalblue'}
    plt.rcParams.update({'mathtext.default': 'regular', 'font.size':10})
    plt.rcParams.update({'text.usetex': False})
    lw=1.5
    if gs==None:
        if args.sort_style == 'all':
            fig = plt.figure(figsize=[5, int(args.n_restore/3)])
            ax = plt.subplot(1,1,1)
        else:
            fig = plt.figure(figsize=[5, int(args.n_tops*args.n_test_params/4)])
            ax = plt.subplot(1,1,1)
    else:
        ax = plt.subplot(gs)

    if 'stat' in args.efficacy_base:
        sbn.boxplot(data=df_top, x='efficacy', y='test_param', order=top_params[args.sort_style], hue='n_test_params', orient='h', 
                saturation=3, width=0.5, whis=2, palette=palette, dodge=False, linewidth=lw, fliersize=2, ax=ax) #fliersize=2

    else:
        sbn.boxplot(data=df_top, x='efficacy', y='test_param', order=top_params[args.sort_style], hue='n_test_params', orient='h', 
                saturation=3, width=0.5, whis=2, palette=palette, dodge=False, linewidth=lw, fliersize=2, ax=ax) #fliersize=2
    
    # data and simulated references
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    

    # Vertical lines for visual indication
    if args.efficacy_base=='sims':
        plt.vlines(0, ymin=ymin, ymax=ymax, linestyle='dashed', color='red', alpha=0.75, linewidth=lw) 
        plt.vlines(100, ymin=ymin, ymax=ymax, linestyle='dashed', color='blue', alpha=0.75, linewidth=lw) 

        plt.vlines((np.std(distances['con_pat'])/np.mean(distances['con_pat_centroid']))*100, ymin=ymin, ymax=ymax, 
                  linestyle='dashed', color='red', alpha=0.5, linewidth=lw)
        plt.vlines((2*np.std(distances['con_pat'])/np.mean(distances['con_pat_centroid']))*100, ymin=ymin, ymax=ymax, 
                   linestyle='dashed', color='red', alpha=0.25, linewidth=lw)

        plt.vlines((1 - (np.std(distances['con'])/np.mean(distances['con_pat_centroid'])))*100, ymin=ymin, ymax=ymax, 
                   linestyle='dashed', color='blue', alpha=0.5, linewidth=lw)
        plt.vlines((1 - (2*np.std(distances['con'])/np.mean(distances['con_pat_centroid'])))*100, ymin=ymin, ymax=ymax, 
                   linestyle='dashed', color='blue', alpha=0.25, linewidth=lw)
    
    elif args.efficacy_base=='paired':
        plt.vlines((1 - (np.mean(distances['con_pat'])/np.mean(distances['con_pat'])))*100, 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='red', alpha=0.75, linewidth=lw) 
        plt.vlines((1 - (np.mean(distances['con'])/np.mean(distances['con_pat'])))*100, 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='blue', alpha=0.75, linewidth=lw) 
        
        plt.vlines((1 - (np.mean(distances['con_pat'])-np.std(distances['con_pat']))/np.mean(distances['con_pat']))*100, ymin=ymin, ymax=ymax, 
                linestyle='dashed', color='red', alpha=0.5, linewidth=lw)
        plt.vlines((1 - (np.mean(distances['con_pat'])-2*np.std(distances['con_pat']))/np.mean(distances['con_pat']))*100, ymin=ymin, ymax=ymax, 
                linestyle='dashed', color='red', alpha=0.25, linewidth=lw)

        plt.vlines((1 - (np.mean(distances['con'])+np.std(distances['con']))/np.mean(distances['con_pat']))*100, ymin=ymin, ymax=ymax, 
                linestyle='dashed', color='blue', alpha=0.5, linewidth=lw)
        plt.vlines((1 - (np.mean(distances['con'])+2*np.std(distances['con']))/np.mean(distances['con_pat']))*100, ymin=ymin, ymax=ymax, 
                linestyle='dashed', color='blue', alpha=0.25, linewidth=lw)
    
    elif args.efficacy_base=='paired_A':
        plt.vlines(1, ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.75, linewidth=lw) 
        
        plt.vlines(((np.mean(distances['con_pat'])+np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.5, linewidth=lw) 
        plt.vlines(((np.mean(distances['con_pat'])-np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.5, linewidth=lw) 
        
        plt.vlines(((np.mean(distances['con_pat'])+2*np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.25, linewidth=lw) 
        plt.vlines(((np.mean(distances['con_pat'])-2*np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.25, linewidth=lw) 

    elif args.efficacy_base=='paired_B':
        plt.vlines(1, ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.75, linewidth=lw) 
        
        plt.vlines((np.mean(distances['con_pat'])/(np.mean(distances['con_pat'])+np.std(distances['con_pat']))), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.5, linewidth=lw) 
        plt.vlines((np.mean(distances['con_pat'])/(np.mean(distances['con_pat'])-np.std(distances['con_pat']))), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.5, linewidth=lw) 
        
        plt.vlines((np.mean(distances['con_pat'])/(np.mean(distances['con_pat'])+2*np.std(distances['con_pat']))), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.25, linewidth=lw) 
        plt.vlines((np.mean(distances['con_pat'])/(np.mean(distances['con_pat'])-2*np.std(distances['con_pat']))), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.25, linewidth=lw) 
        
    elif args.efficacy_base=='paired_C':
        plt.vlines(0, ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.75, linewidth=lw) 
        
        plt.vlines((1-(np.mean(distances['con_pat'])+np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.5, linewidth=lw) 
        plt.vlines((1-(np.mean(distances['con_pat'])-np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.5, linewidth=lw) 
        
        plt.vlines((1-(np.mean(distances['con_pat'])+2*np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.25, linewidth=lw) 
        plt.vlines((1-(np.mean(distances['con_pat'])-2*np.std(distances['con_pat']))/np.mean(distances['con_pat'])), 
                ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.25, linewidth=lw) 
    
    elif args.efficacy_base=='tstat':
        plt.vlines(5, ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.75, linewidth=lw) 

    elif args.efficacy_base=='ustat':
        sig_ustats = df_restore[(df_restore.upval*1485<0.051) & (df_restore.upval*1485>0.049)].ustat.unique()
        plt.vlines(np.abs(sig_ustats).mean()/(400*400), ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.75, linewidth=lw) 

    else:
        plt.vlines(0, ymin=ymin, ymax=ymax, linestyle='dashed', color='gray', alpha=0.75, linewidth=lw) 
    
    # legends and labels 
    labels = plt.gca().get_yticklabels()
    new_labels = format_labels(labels)
    ax.set_yticklabels(new_labels)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.set_xlabel("$Efficacy \quad (E_{ff}, \\ \%)$", fontsize=10)
    if 'tstat' in args.efficacy_base:
            ax.set_xlabel("$T \, statistic$", fontsize=10)
    if 'ustat' in args.efficacy_base:
            ax.set_xlabel("AUC", fontsize=10)
            #ticks = ax.get_xticks()
            #ax.set_xticklabels(labels=["{:.1f}".format(i/100000) for i in ticks])
    ax.set_ylabel("Target points", fontsize=10)
    
    sbn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.get_legend().set_visible(False)
    
    if ((gs==None) and (args.save_figs)):
        fname = 'restoration_best_efficacy_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))

    #plt.show(block=False)


def plot_efficacy_by_number_of_target(df_top, gs=None, args=None):
    """ Swarm plots of efficacy score (y-axis) by number of targets (x-axis), with means projected in log-linear scale. 
    
    Parameters
    ----------
        df_top: pandas.DataFrame
            Subset of df_restore with only virtual intervention resulting in positive outcomes.
        gs: matplotlib.GridSpec
            (optional) A GridSpec object that can be used to embbed axes when this figure is a subplot 
            of a larger figure. 
        args: argparse.Namespace
            Extra arguments with options. 

    """
    palette = {0: 'white', 1: 'lightpink', 2: 'plum', 3: 'mediumpurple', 4: 'lightsteelblue', 5:'skyblue', 6:'royalblue'}
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    if gs==None:
        fig = plt.figure(figsize=[500*px,200*px])
        gsp = plt.GridSpec(nrows=1, ncols=5)
        ax1 = plt.subplot(gsp[0,0:3])
        ax2 = plt.subplot(gsp[0,3:])
    else:
        ax1 = plt.subplot(gs[0,0:3])
        ax2 = plt.subplot(gs[0,4])

    plt.sca(ax1)
    plt.tight_layout()
    if args.use_optim_params:
        sbn.swarmplot(data=df_top, x='n_test_params', y='efficacy', ax=ax1, size=2, palette=palette, alpha=0.6)
    #else:
        #sbn.stripplot(data=df_top, x='n_test_params', y='efficacy', ax=ax1, size=0.5, palette=palette, alpha=0.6)
    sbn.boxplot(data=df_top, x='n_test_params', y='efficacy', ax=ax1, width=0.1, palette=palette, fliersize=0, linewidth=1.5, showcaps=False)

    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)

    plt.xlabel("$n_t$", fontsize=10)

    ticks = ax1.get_yticks()
    #ax1.set_yticklabels(labels=["{:.1f}".format(i/100000) for i in ticks])
    plt.ylabel("AUC", fontsize=10)


    plt.sca(ax2)
    plt.tight_layout()
    lines = []
    for i,j in enumerate(np.sort(df_top.n_test_params.unique())):
        if i==0:
            mu_t0 = 0
        else:
            mu_t0 = df_top[df_top.n_test_params==i].efficacy.mean()
        mu_t1 = df_top[df_top.n_test_params==j].efficacy.mean()
        #plt.plot(j, np.log(mu_t1-mu_t0), 'o', color=palette[j], ms=9)
        #lines.append({'x':j, 'y':np.log(mu_t1-mu_t0)})
        #plt.plot(j, np.log10(mu_t1), 'o', color=palette[j], ms=9)
        plt.plot(np.log(j), mu_t1, 'o', color=palette[j], ms=6)
        lines.append({'x':np.log(j), 'y':mu_t1})
    sbn.regplot(data=pd.DataFrame(lines), x='x', y='y', ci=95, color='gray', ax=ax2)
    ax2.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.set_xticks(np.log(np.arange(1,7)))
    ax2.set_xticklabels(np.arange(1,7))
    #ax.set_xlim([0.5,6.5])
    #ax.set_xscale('log')
    plt.xlabel("$n_t$", fontsize=10)
    #ticks = ax2.get_yticks()
    #ax2.set_yticklabels(labels=["{:.1f}".format(i/100000) for i in ticks]) 
    #plt.ylabel("$U \; statistic \; ( x 10^5)$", fontsize=10)
    plt.ylabel("$\widehat{AUC}$", fontsize=10)

    if gs==None:
        plt.tight_layout()
        if args.save_figs:
            fname = 'avg_efficacy_by_n_targets_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
            plt.savefig(os.path.join(proj_dir, 'img', fname))



#-----------------------------#
# FEATURE IMPORTANCE ANALYSIS #
#-----------------------------#

def get_X_y(df_restore, params):
    """ 
    Extract parameter features for hierarchical clustering of test parameters in restoration analysis 

    Parameters:
    -----------
    df_restore: pandas DataFrame 
        Simulation data with a test_param and a distance column to be transformed to X and y
    params: list
        Parameters of model

    Results:
    --------
    X: np.array
        input feature vector 
    y: np.array
        output vector
    """
    # template feature set all feature params to 0
    template_features = dict((par,0) for par in params)

    def transform_test_param(test_param):
        """ return a binary array of param features """
        out = copy.deepcopy(template_features)
        for par in test_param.split(' '):
            out[par] = 1
        return list(out.values())

    X = df_restore.test_param.apply(transform_test_param)
    y = df_restore['efficacy']

    return np.array(list(X)), np.array(y)


def decision_tree(df_restore, params, args):
    """ create a tree from restoration output 
    Parameters:
    -----------
        df_restore
            pandas DataFrame of restoration output
        params
            list of model parameters
        args: argparse Namespace
            global arguments, used to get number of tested parameters
    """
    # create a line in df for each number of params tested
    lines = [] 
    decision_trees = []
    feat_imps = dict() # feature importances
    for n_test_params in np.arange(args.n_test_params)+1:
        X,y = get_X_y(df_restore[(df_restore.n_test_params==n_test_params) & (df_restore.efficacy>0)], params)
        #X,y = get_X_y(df_restore, params)
        dt = sklearn.tree.DecisionTreeRegressor(max_depth=args.max_depth)
        dt.fit(X, y)
        y_pred = dt.predict(X)
        decision_trees.append({'dt':dt, 'y':y, 'y_pred':y_pred})

        # basic feature importances
        line = dict((feat, imp) for feat,imp in zip(params, dt.feature_importances_))
        line['n_test_params'] = n_test_params
        lines.append(line)

        # more detailed feature importances
        feat_imps[n_test_params] = permutation_importance(dt, X, y, scoring='neg_mean_absolute_error', n_repeats=10)

    df_feature_importance = pd.DataFrame(lines)
    return df_feature_importance, decision_trees, feat_imps
        
def plot_decision_tree(df_dt, decision_trees, params):
    """ Plot a bar plot of basic feature importance and a tree diagram """
    plt.rcParams.update({'font.size':12})
    for i,row in df_dt.iterrows():
        plt.figure(figsize=[6,4])
        sbn.barplot(row[params].to_frame().transpose())
        plt.xticks(rotation=30)
        plt.show()

        plt.figure(figsize=[60,20])
        sklearn.tree.plot_tree(decision_trees[i]['dt'], max_depth=6, feature_names=params, proportion=True, impurity=False)
        plt.show()
    

def plot_feature_importances(feat_imps, params):
    """ Plot feature importance using permutations (not bulit-in basic) """
    fig = plt.figure(figsize=[16,6])
    for i,(n_test_params, feat_imp) in enumerate(feat_imps.items()):
        fi = dict()
        for feat, imp in zip(params, feat_imp.importances_mean):
            fi[feat] = imp
        df_featimp = pd.DataFrame([fi])

        ax = plt.subplot(2,3,i+1)
        sbn.barplot(data=df_featimp, ax=ax)
        plt.title(f'n_test_params = {n_test_params}')
        plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_dt_prediction(decision_trees):
    """ Plot decision tree prediction vs ground truth for sanity check """
    n_dts = len(decision_trees)
    fig = plt.figure(figsize=[n_dts*5, 5])
    for i,dt in enumerate(decision_trees):
        plt.subplot(1,n_dts, i+1)
        plt.scatter(dt['y'], dt['y_pred'], alpha=0.005, s=10)
        plt.xlabel('y')
        plt.ylabel('y_pred')
        r,p = scipy.stats.pearsonr(dt['y'], dt['y_pred'])
        r2 = r2_score(dt['y'], dt['y_pred'])
        plt.title('r={:.2f}  p={:.3f}     r2={:.3f}'.format(r,p,r2))
    plt.show()


def plot_feature_windrose(df_dt, params, rscale='linear', args=None):
    """ Make windrose vizualisation of decision tree feature importances """
    theta = np.arange(len(params))/(len(params)) * 2*np.pi
    theta = np.append(theta, theta[0])

    palette = {0: 'white', 1: 'lightpink', 2: 'plum', 3: 'mediumpurple', 4: 'lightsteelblue', 5:'skyblue', 6:'royalblue'}

    fig, axes = plt.subplots(3,2, subplot_kw={'projection': 'polar'}, figsize=[5,8])
    for i in np.arange(len(df_dt)):
        j = int(np.floor(i/2))
        k = i%2
        r = np.array(df_dt.iloc[i][params])
        #r = np.array(feat_imps[i+1].importances_mean)
        
        #rmin = r - np.array(feat_imps[i+1].importances_std)/2
        #rmin[rmin<0] = 0
        #rmax = r + np.array(feat_imps[i+1].importances_std)/2
        
        r = np.append(r, r[0]) 
        #rmin = np.append(rmin, rmin[0]) 
        #rmax = np.append(rmax, rmax[0]) 

        #axes[i].plot(theta, r, label = str(df_dt.iloc[i].n_test_params), color=palette[df_dt.iloc[i].n_test_params], lw=5)
        #axes[j,k].bar(theta_, np.abs(r_), label = str(df_dt.iloc[i].n_test_params), color=palette[df_dt.iloc[i].n_test_params], lw=5, width=0.5)
        for r_, theta_ in zip(r,theta):
            if r_<0:
                r__ = np.log10(np.abs(r_)) if rscale=='log' else np.abs(r_)
                axes[j,k].bar(theta_, r__, facecolor=palette[df_dt.iloc[i].n_test_params], lw=1, linestyle='--', width=0.5, edgecolor='black')
            else:
                r__ = np.log10(r_) if rscale=='log' else r_
                axes[j,k].bar(theta_, r__, facecolor=palette[df_dt.iloc[i].n_test_params], lw=1, linestyle='-', width=0.5, edgecolor='black')
        #axes[i].fill_between(theta, rmin, rmax)
        axes[j,k].set_xticks(theta[:-1])
        axes[j,k].set_xticklabels(params)
        lbls = axes[j,k].get_xticklabels()
        new_lbls = format_labels(lbls)
        axes[j,k].set_xticklabels(new_lbls)
        #axes[i].set_rticks([0.1], labels=[])
        axes[j,k].set_yticks([-1,0,1,2])
        axes[j,k].set_yticklabels([])
        #axes[j,k].set_rmax(2)
        #axes[j,k].set_rmin(-2)
        axes[j,k].spines.polar.set_visible(False)
        axes[j,k].xaxis.grid(linewidth=0.5, linestyle='--')

    plt.tight_layout()
    if args.save_figs:
        #plt.rcParams['svg.fonttype'] = 'none'
        fname = 'restoration_param_importances_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))
    plt.show()


def compute_dt_best_paths(decision_tree, args):
    """ Find best path and extract corresponding features 

        Returns
        -------
            None. The paths are added in given decision_tree dictionary
    """
    dt = decision_tree['dt']
    n_bests = 5
    best_leaf = np.argsort(dt.tree_.value.squeeze() * np.array(dt.tree_.feature==-2, dtype=int))[::-1][:n_bests]
    depth = dt.tree_.compute_node_depths()[best_leaf]

    def get_parent(node_id):
        for node in range(dt.tree_.node_count):
            if dt.tree_.children_left[node]==node_id:
                return {node:False}
            elif dt.tree_.children_right[node]==node_id:
                return {node:True}
        return {}
    
    # get list of parents
    parents = []
    for i,p in enumerate(best_leaf):
        branch = []
        for _ in range(depth[i]):
            parent = get_parent(p)
            if parent!={}:
                branch.append(parent)
                p = list(parent.keys())[0]
        parents.append(branch)
    
    decision_tree['paths'] = parents
    decision_tree['best_leaves'] = best_leaf    


def get_feat_scores(decision_tree, params):
    """ 
    Compute custom scores of features 
    
    Returns:
    --------
    feat_scores: dict
        (key,value) = (feature_name, feature_score)
    """
    dt = decision_tree['dt']
    parents = decision_tree['paths']
    best_leaves = decision_tree['best_leaves']
    pars = np.array(params)
    feat_scores = dict((param,0) for param in params)
    for i,branch in enumerate(parents):
        for node in branch:
            if list(node.values())[0]:
                node_id = list(node.keys())[0]
                feat_name = pars[dt.tree_.feature[node_id]]
                score = dt.tree_.value[best_leaves[i]].squeeze()
                feat_scores[feat_name] += score
    return feat_scores


def compute_custom_feature_scores(decision_trees, params, args):
    lines= [] 
    for i,decision_tree in enumerate(decision_trees):
        compute_dt_best_paths(decision_tree, args)
        feat_scores = get_feat_scores(decision_tree, params)
        feat_scores['n_test_params'] = i+1
        lines.append(feat_scores)
    df_custom_feat_imps = pd.DataFrame(lines)
    return df_custom_feat_imps


def compute_simple_feature_scores(df_top, params, args):
    """ Compute feature score simply based on top params weighted by efficacy """
    lines = []
    for n_test_params in np.arange(args.n_test_params)+1:
        df_ = df_top[df_top.n_test_params==n_test_params]
        Xs, ys = get_X_y(df_, params)
        feat_imp = np.zeros((len(params),))
        for X,y in zip(Xs, ys):
            feat_imp += np.array(X*y).squeeze()
        line = dict((feat,imp) for feat,imp in zip(params,feat_imp))
        line['n_test_params'] = n_test_params
        lines.append(line)
    df_simple_feat_imps = pd.DataFrame(lines)
    return df_simple_feat_imps


def scale_efficacy_to_kdes(df_row, params, kdes, scaling):
    normalized = dict()
    for param in params:
        #val = (df_row[param] - kdes['patients'][param]['vals'].mean()) / kdes['patients'][param]['vals'].std()
        if scaling=='contribution':
            #normalized[param] = df_row['efficacy']*val
            normalized[param] = df_row['efficacy']*df_row['z_'+param]
        else: # sensitivity
            #normalized[param] = df_row['efficacy']/val
            normalized[param] = df_row['efficacy']/df_row['z_'+param]
    return normalized


def compute_scaled_feature_score(df_top, params, kdes, scaling='dot_product_correlation', args=None):
    """ Compute feature scores (i.e. parameter contribution) as the dot-product between their normalized location on the 
    KDEs distribution (using z-statistic) and their resultng efficacy (AUC). 
    
    Parameters
    ----------
        df_top: pandas.DataFrame
            Subset of df_restore with significantly positive virtual interventions
        params: list
            Individual intervention targets (i.e. model parameters).
        kdes: dict
            Kernel Density Estimates of posterior distributions of OCD subjects and healthy controls.
        scaling: string
            How to scale the efficacy of the virtual intervention by the z-score normalized parameter. 
            "dot_product_correlation" (default) multiplies the normalized parameter by the efficacy (AUC). 
            Other values can be "pearson_correlation", "spearman_correlation" and "covariance_correlation" 
            but those measures distort the results and interpretation due to the mean-centering of variable.
            Other legacy values are "contribution" (same as dot-product) and "sensitivity" which divides the 
            normalized parameter by the AUC efficacy (giving a sense of "sensitivity" of the parameter). 

        args: argparse.Namespace
            (optional) Extra arguments with options. 

    """
    lines = []
    for n_test_params in np.arange(args.n_test_params)+1:
        df_ = df_top[df_top.n_test_params==n_test_params]
        print("Compute param sensitivity for n_test_params={}".format(n_test_params))
        if 'correlation' not in scaling:
            rows = []
            for i,row in df_.iterrows():
                new_row = scale_efficacy_to_kdes(row, params, kdes, scaling)
                rows.append(new_row)
            line = pd.DataFrame(rows).mean(axis=0).to_frame().transpose()
            line['n_test_params'] = n_test_params
            lines.append(line)
        else:
            line = dict()
            line['n_test_params'] = n_test_params
            diff_fc = np.array(df_['dist_pre_hc']) - np.array(df_['dist'])
            for param in params:
                inds, = np.where([param in test_param.split(' ') for test_param in df_.test_param])
                if scaling=='pearson_correlation':
                    R,p = scipy.stats.pearsonr(diff_fc, np.array(df_['z_'+param]))
                    line[param] = R
                    line['p_'+param] = p
                elif scaling=='spearman_correlation':
                    R,p = scipy.stats.spearmanr(diff_fc, np.array(df_['z_'+param]))
                    line[param] = R
                    line['p_'+param] = p    
                elif scaling == 'covariance_correlation':
                    R = np.cov(diff_fc, np.array(df_['z_'+param]))[0,1]
                    line[param] = R
                    line['p_'+param] = None
                elif scaling == 'cross_correlation':
                    R = np.correlate(diff_fc, np.array(df_['z_'+param]))[0]
                    line[param] = R
                    R = R / (np.std(diff_fc)*np.std(df_['z_'+param])) / len(diff_fc)
                    line['R_'+param] = R
                    
            line['n'] = len(df_)
            lines.append(pd.DataFrame([line]))
    df_kdes_scaled_feats = pd.concat(lines, ignore_index=True)
    return df_kdes_scaled_feats


def compute_feature_reliability(df_top, params, kdes, args=None):
    """ Compute reliability of feature using z-score """
    out = []
    for n_test_params in np.arange(args.n_test_params)+1:
        df_ = df_top[df_top.n_test_params==n_test_params]
        print("Compute param reliability for n_test_params={}".format(n_test_params))
        for param in params:
            df_['z_'+param] = (df_[param] - kdes['patients'][param]['vals'].mean()) / kdes['patients'][param]['vals'].std()
        out.append(df_)
    df_reliability = pd.concat(out)
    return df_reliability


def plot_contribution_windrose(df_params_contribution, params, args=None):
    """ Make windrose vizualisation of (normalized) parameter contribution in log-scaled """
    theta = np.arange(len(params))/(len(params)) * 2*np.pi
    theta = np.append(theta, theta[0])

    palette = {0: 'white', 1: 'lightpink', 2: 'plum', 3: 'mediumpurple', 4: 'lightsteelblue', 5:'skyblue', 6:'royalblue'}

    fig, axes = plt.subplots(3,2, subplot_kw={'projection': 'polar'}, figsize=[4,6])
    for i in np.arange(len(df_params_contribution)):
        j = int(np.floor(i/2))
        k = i%2
        r = np.array(df_params_contribution.iloc[i][params])
        r = np.append(r, r[0]) 
        
        for r_, theta_ in zip(r,theta):
            if r_<0:
                axes[j,k].bar(theta_, np.abs(r_), facecolor=palette[df_params_contribution.iloc[i].n_test_params], lw=0.75, linestyle='--', width=0.5, edgecolor='black', zorder=3)
            else:
                axes[j,k].bar(theta_, r_, facecolor=palette[df_params_contribution.iloc[i].n_test_params], lw=0.75, linestyle='-', width=0.5, edgecolor='black', zorder=3)
        
        axes[j,k].set_xticks(theta[:-1])
        axes[j,k].set_xticklabels(args.params)
        lbls = axes[j,k].get_xticklabels()
        new_lbls = format_labels(lbls)
        axes[j,k].set_xticklabels(new_lbls, fontsize=8)
        
        r_max = np.max([np.abs(r).max(), 101])
        axes[j,k].set_yticks(np.arange(0,r_max, 100))
        axes[j,k].set_yticklabels([])
        #axes[j,k].set_rmax(2.2)
        axes[j,k].grid(zorder=0)
        
        axes[j,k].spines.polar.set_visible(False)
        axes[j,k].xaxis.grid(linewidth=0.5, linestyle='--')

    plt.tight_layout()
    if args.save_figs:
        #plt.rcParams['svg.fonttype'] = 'none'
        fname = 'normalized_params_contribution_'+args.distance_metric+'_'+args.efficacy_base+today()+'_small.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))
    plt.show()

    
def plot_single_contribution_windrose(df, params, theta, palette, ax):
    """ A single windrose of parameters' contribution. 
    
    Parameters
    ----------
        df: pandas.DataFrame 
            Contribution data.
        theta: list
            Angles at which to put polar bars. 
        palette: dict
            Color palette used by matplotlib.
        ax: matplotlib.Axes 
            Axes to plot the windrose (must be polar type).

    """ 
    r = np.array(df[params])
    r = np.append(r, r[0]) 
    
    for r_, theta_ in zip(r,theta):
        if r_<0:
            ax.bar(theta_, np.abs(r_), facecolor=palette[df['n_test_params'].iloc[0]], lw=0.75, linestyle='--', width=0.5, edgecolor='black', zorder=3)
        else:
            ax.bar(theta_, r_, facecolor=palette[df['n_test_params'].iloc[0]], lw=0.75, linestyle='-', width=0.5, edgecolor='black', zorder=3)
    
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(params)
    lbls = ax.get_xticklabels()
    new_lbls = format_labels(lbls)
    ax.set_xticklabels(new_lbls, fontsize=10)
    
    #r_max = np.max([np.abs(r).max(), 1])
    #ax.set_yticks(np.arange(0,r_max, 1))
    #r_max = np.max([np.abs(r).max(), 101])
    #ax.set_yticks(np.arange(0,r_max, 100))
    #r_max = np.max([np.abs(r).max(), 1.1])
    #ax.set_yticks(np.arange(0,r_max, 1))
    ax.set_yticks([])
    ax.set_yticklabels([])
    #ax.set_rmax(2.2)
    ax.grid(zorder=0)
    
    ax.spines.polar.set_visible(False)
    ax.xaxis.grid(linewidth=0.5, linestyle='--')


def plot_parameters_contribution(df_params_contribution, params, gs=None, args=None):
    """ Polar plots of parameters contribution across virtual interventions, colorcoded by number of intervention targets :math:`n_t`.
    Each polar plot corresponds to a number of target :math:`n_t`. 
    
    Parameters
    ----------
        df_params_contribution: pandas.DataFrame
            Contribution of parameters (z-score normalized parameters times efficacies of virtual interventions).
        params: list
            Individual intervention targets (i.e. model parameters).
        gs: matplotlib.GridSpec
            (optional) A GridSpec object that can be used to embbed axes when this figure is a subplot of a larger figure.
        args: argparse.Namespace
            (optional) Extra arguments with options. 
    
    """ 
    theta = np.arange(len(params))/(len(params)) * 2*np.pi
    theta = np.append(theta, theta[0])

    palette = {0: 'white', 1: 'lightpink', 2: 'plum', 3: 'mediumpurple', 4: 'lightsteelblue', 5:'skyblue', 6:'royalblue'}

    if gs==None:
        #fig, axes = plt.subplots(3,2, subplot_kw={'projection': 'polar'}, figsize=[4,6])
        fig = plt.figure(figsize=[4,6])
        gs = plt.GridSpec(nrows=3, ncols=2)

    for i,n_test_params in enumerate(np.sort((df_params_contribution.n_test_params.unique()))):
        df = df_params_contribution[df_params_contribution.n_test_params==n_test_params]
        ax = plt.subplot(gs[i//2,i%2], polar=True)
        plot_single_contribution_windrose(df, params, theta, palette, ax=ax)

    plt.tight_layout()
    

def plot_sensitivity_windrose(df_params_sensitivity, params, args=None):
    """ Make windrose vizualisation of (normalized) parameter sensitivity in log-scale """
    theta = np.arange(len(params))/(len(params)) * 2*np.pi
    theta = np.append(theta, theta[0])

    palette = {0: 'white', 1: 'lightpink', 2: 'plum', 3: 'mediumpurple', 4: 'lightsteelblue', 5:'skyblue', 6:'royalblue'}

    fig, axes = plt.subplots(3,2, subplot_kw={'projection': 'polar'}, figsize=[5,8])
    for i in np.arange(len(df_params_sensitivity)):
        j = int(np.floor(i/2))
        k = i%2
        r = np.array(df_params_sensitivity.iloc[i][params])
        r = np.append(r, r[0]) 
        
        for r_, theta_ in zip(r,theta):
            if r_<0:
                axes[j,k].bar(theta_, np.log10(np.abs(r_)), facecolor=palette[df_params_sensitivity.iloc[i].n_test_params], lw=0, linestyle='--', width=0.5, edgecolor='black')
            else:
                axes[j,k].bar(theta_, np.log10(r_), facecolor=palette[df_params_sensitivity.iloc[i].n_test_params], lw=0, linestyle='-', width=0.5, edgecolor='black')
        
        axes[j,k].set_xticks(theta[:-1])
        axes[j,k].set_xticklabels(params)
        lbls = axes[j,k].get_xticklabels()
        new_lbls = format_labels(lbls)
        axes[j,k].set_xticklabels(new_lbls)
        
        axes[j,k].set_yticks([0,1,2,3])
        axes[j,k].set_yticklabels([])
        axes[j,k].set_rmax(3.5)

        axes[j,k].spines.polar.set_visible(False)
        axes[j,k].xaxis.grid(linewidth=0.5, linestyle='--')

    plt.tight_layout()
    if args.save_figs:
        #plt.rcParams['svg.fonttype'] = 'none'
        fname = 'normalized_params_log_sensitivity_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))
    plt.show()


#---------------------#
# PREDICTIVE ANALYSIS #
#---------------------#

def plot_fc_dist_pre_post_behav(df_summary, feature='dist', args=None):
    """ plot behavioral relationship to distance to FC controls """
    #behav = 'YBOCS_Total' #'OCIR_Total'
    behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total', 'Dep_Total', 'Anx_total']
    colors={'group1':'orange', 'group2':'green'}

    plt.figure(figsize=[21,5])
    for i,behav in enumerate(behavs):
        plt.subplot(1,len(behavs), i+1)
        feat_diffs = {'group1':[], 'group2':[], 'both':[]}
        behav_diffs = {'group1':[], 'group2':[], 'both':[]}
        responders = {feature:copy.deepcopy(feat_diffs), 'behav':copy.deepcopy(behav_diffs)}
        for subj in df_summary.subj.unique():
            df_subj = df_summary[df_summary.subj==subj]
            group = df_subj['group'].unique()[0]
            diff = df_subj[df_subj.ses=='ses-post'][feature].iloc[0] - df_subj[df_subj.ses=='ses-pre'][feature].iloc[0]
            behav_diff = df_subj[df_subj.ses=='ses-post'][behav].iloc[0] - df_subj[df_subj.ses=='ses-pre'][behav].iloc[0]
            feat_diffs[group].append(diff)
            behav_diffs[group].append(behav_diff)
            feat_diffs['both'].append(diff)
            behav_diffs['both'].append(behav_diff)
            if diff < 0:
                plt.scatter(behav_diff, diff, color=colors[group], alpha=0.5)
                responders[feature][group].append(diff)
                responders['behav'][group].append(behav_diff)
                responders[feature]['both'].append(diff)
                responders['behav']['both'].append(behav_diff)
            else:
                plt.scatter(behav_diff, diff, color=colors[group], alpha=0.2)

        r,p = scipy.stats.pearsonr(feat_diffs['both'], behav_diffs['both'])
        r1,p1 = scipy.stats.pearsonr(feat_diffs['group1'], behav_diffs['group1'])
        r2,p2 = scipy.stats.pearsonr(feat_diffs['group2'], behav_diffs['group2'])
        rr,pr = scipy.stats.pearsonr(responders[feature]['both'], responders['behav']['both'])
        r1r,p1r = scipy.stats.pearsonr(responders[feature]['group1'], responders['behav']['group1'])
        r2r,p2r = scipy.stats.pearsonr(responders[feature]['group2'], responders['behav']['group2'])
        plt.title("n={}  r={:.2f}  p={:.3f}\ng1: n={} r={:.2f} p={:.3f}\ng2: n={} r={:.2f} p={:.3f}\n\
            Responders:\n n={}  r={:.2f}  p={:.3f}\ng1: n={} r={:.2f} p={:.3f}\ng2: n={} r={:.2f} p={:.3f}".format(
            len(feat_diffs['both']),r,p, len(feat_diffs['group1']), r1, p1, len(feat_diffs['group2']), r2, p2,
            len(responders[feature]['both']),rr,pr, len(responders[feature]['group1']), r1r, p1r, len(responders[feature]['group2']), r2r, p2r))
        plt.xlabel("$\Delta \, {}$".format(behav.split('_')[0]))
        plt.ylabel("$\Delta \, {} \, (FC)$".format(feature))
        plt.gca().spines.top.set_visible(False)
        plt.gca().spines.right.set_visible(False)
    plt.tight_layout()


def plot_fc_dist_pre_post_params(df_summary, args=None):
    """ plot behavioral relationship to distance to FC controls """
    #behav = 'YBOCS_Total' #'OCIR_Total'
    params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42', 'eta_C_13', 'eta_C_24', 'sigma', 'sigma_C_13', 'sigma_C_24']
    colors={'group1':'orange', 'group2':'green'}

    plt.figure(figsize=[18,10])
    for i,param in enumerate(params):
        plt.subplot(2,int(np.ceil(len(params)/2)), i+1)
        dist_diffs = {'group1':[], 'group2':[], 'both':[]}
        param_diffs = {'group1':[], 'group2':[], 'both':[]}
        responders = {'dist':copy.deepcopy(dist_diffs), 'param':copy.deepcopy(param_diffs)}
        for subj in df_summary.subj.unique():
            df_subj = df_summary[df_summary.subj==subj]
            group = df_subj['group'].unique()[0]
            diff = df_subj[df_subj.ses=='ses-post'].dist.iloc[0] - df_subj[df_subj.ses=='ses-pre'].dist.iloc[0]
            param_diff = df_subj[df_subj.ses=='ses-post'][param].iloc[0] - df_subj[df_subj.ses=='ses-pre'][param].iloc[0]
            dist_diffs[group].append(diff)
            param_diffs[group].append(param_diff)
            dist_diffs['both'].append(diff)
            param_diffs['both'].append(param_diff)
            if diff < 0:
                plt.scatter(param_diff, diff, color=colors[group], alpha=0.5)
                responders['dist'][group].append(diff)
                responders['param'][group].append(param_diff)
                responders['dist']['both'].append(diff)
                responders['param']['both'].append(param_diff)
            else:
                plt.scatter(param_diff, diff, color=colors[group], alpha=0.2)

        r,p = scipy.stats.pearsonr(dist_diffs['both'], param_diffs['both'])
        r1,p1 = scipy.stats.pearsonr(dist_diffs['group1'], param_diffs['group1'])
        r2,p2 = scipy.stats.pearsonr(dist_diffs['group2'], param_diffs['group2'])
        rr,pr = scipy.stats.pearsonr(responders['dist']['both'], responders['param']['both'])
        r1r,p1r = scipy.stats.pearsonr(responders['dist']['group1'], responders['param']['group1'])
        r2r,p2r = scipy.stats.pearsonr(responders['dist']['group2'], responders['param']['group2'])
        plt.title("n={}  r={:.2f}  p={:.3f}\ng1: n={} r={:.2f} p={:.3f}\ng2: n={} r={:.2f} p={:.3f}\n\
            Responders:\n n={}  r={:.2f}  p={:.3f}\ng1: n={} r={:.2f} p={:.3f}\ng2: n={} r={:.2f} p={:.3f}".format(
            len(dist_diffs['both']),r,p, len(dist_diffs['group1']), r1, p1, len(dist_diffs['group2']), r2, p2,
            len(responders['dist']['both']),rr,pr, len(responders['dist']['group1']), r1r, p1r, len(responders['dist']['group2']), r2r, p2r))
        plt.xlabel("$\Delta \, {}$".format(format_param(param)))
        plt.ylabel("$\Delta \, distance \, (FC)$")
        plt.gca().spines.top.set_visible(False)
        plt.gca().spines.right.set_visible(False)
    plt.tight_layout()



def plot_pre_post_params_behavs(df_summary, args=None):
    """ plot behavioral relationship to distance to FC controls """

    params=['dist', 'efficacy', 'C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42', 'eta_C_13', 'eta_C_24', 'sigma', 'sigma_C_13', 'sigma_C_24']
    behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total', 'Dep_Total', 'Anx_total']
    colors={'group1':'orange', 'group2':'green'}

    fig = plt.figure(figsize=[len(behavs)*3,len(params)*5])
    gs = plt.GridSpec(nrows=len(params), ncols=len(behavs))
    for i,param in enumerate(params):
        for j,behav in enumerate(behavs):
            ax = fig.add_subplot(gs[i,j])
            behav_diffs = {'group1':[], 'group2':[], 'both':[]}
            param_diffs = {'group1':[], 'group2':[], 'both':[]}
            responders = {'behav':copy.deepcopy(behav_diffs), 'param':copy.deepcopy(param_diffs)}
            for subj in df_summary.subj.unique():
                df_subj = df_summary[df_summary.subj==subj]
                group = df_subj['group'].unique()[0]
                behav_diff = df_subj[df_subj.ses=='ses-post'][behav].iloc[0] - df_subj[df_subj.ses=='ses-pre'][behav].iloc[0]
                param_diff = df_subj[df_subj.ses=='ses-post'][param].iloc[0] - df_subj[df_subj.ses=='ses-pre'][param].iloc[0]
                behav_diffs[group].append(behav_diff)
                param_diffs[group].append(param_diff)
                behav_diffs['both'].append(behav_diff)
                param_diffs['both'].append(param_diff)
                if behav_diff < -1:
                    plt.scatter(param_diff, behav_diff, color=colors[group], alpha=0.5)
                    responders['behav'][group].append(behav_diff)
                    responders['param'][group].append(param_diff)
                    responders['behav']['both'].append(behav_diff)
                    responders['param']['both'].append(param_diff)
                else:
                    plt.scatter(param_diff, behav_diff, color=colors[group], alpha=0.2)

            r,p = scipy.stats.pearsonr(behav_diffs['both'], param_diffs['both'])
            r1,p1 = scipy.stats.pearsonr(behav_diffs['group1'], param_diffs['group1'])
            r2,p2 = scipy.stats.pearsonr(behav_diffs['group2'], param_diffs['group2'])
            rr,pr = scipy.stats.pearsonr(responders['behav']['both'], responders['param']['both'])
            r1r,p1r = scipy.stats.pearsonr(responders['behav']['group1'], responders['param']['group1'])
            r2r,p2r = scipy.stats.pearsonr(responders['behav']['group2'], responders['param']['group2'])
            plt.title("n={}  r={:.2f}  p={:.3f}\ng1: n={} r={:.2f} p={:.3f}\ng2: n={} r={:.2f} p={:.3f}\n\
                Responders:\n n={}  r={:.2f}  p={:.3f}\ng1: n={} r={:.2f} p={:.3f}\ng2: n={} r={:.2f} p={:.3f}".format(
                len(behav_diffs['both']),r,p, len(behav_diffs['group1']), r1, p1, len(behav_diffs['group2']), r2, p2,
                len(responders['behav']['both']),rr,pr, len(responders['behav']['group1']), r1r, p1r, len(responders['behav']['group2']), r2r, p2r))
            plt.xlabel("$\Delta \, {}$".format(format_param(param)))
            plt.ylabel("$\Delta \, {}$".format(behav.split('_')[0]))
            plt.gca().spines.top.set_visible(False)
            plt.gca().spines.right.set_visible(False)
    plt.tight_layout()


def plot_pre_post_dist_ybocs(df_summary, gs=None, args=None):
    """ Plot improvement in behavioral measure of symptoms severity (Y-BOCS) of subjects, and their association to 
    functional improvement (via their distance to healthy functional connectivity).
    
    Parameters
    ----------
        df_summary: pandas.DataFrame
            Summary measures of empirical analysis, inluding subject's functional connectivity across frontostriatal circuits,
            behavioral measures (e.g. Y-BOCS score, IQ, etc.), digital pairing and values of the digital twin parameters.
        gs: matplotlib.Gridspec
            (optional) A GridSpec object that can be used to embbed axes when this figure is a subplot 
            of a larger figure.  
        args: argparse.Namespace
            Extra arguments with options. 
    
    """
    param = 'dist'
    behav = 'YBOCS_Total'
    if gs==None:
        fig = plt.figure(figsize=[3,3])
        ax = fig.add_subplot(1,1,1)
    else:
        ax = plt.subplot(gs)

    behav_diffs = {'group1':[], 'group2':[], 'both':[]}
    param_diffs = {'group1':[], 'group2':[], 'both':[]}
    responders = {'behav':copy.deepcopy(behav_diffs), 'param':copy.deepcopy(param_diffs)}
    lines=[]
    for subj in df_summary.subj.unique():
        df_subj = df_summary[df_summary.subj==subj]
        group = df_subj['group'].unique()[0]
        behav_diff = df_subj[df_subj.ses=='ses-pre'][behav].iloc[0] - df_subj[df_subj.ses=='ses-post'][behav].iloc[0]
        param_diff = df_subj[df_subj.ses=='ses-pre'][param].iloc[0] - df_subj[df_subj.ses=='ses-post'][param].iloc[0]
        behav_diffs[group].append(behav_diff)
        param_diffs[group].append(param_diff)
        behav_diffs['both'].append(behav_diff)
        param_diffs['both'].append(param_diff)
        
        responders['behav'][group].append(behav_diff)
        responders['param'][group].append(param_diff)
        responders['behav']['both'].append(behav_diff)
        responders['param']['both'].append(param_diff)
        lines.append({'subj':subj, 'group':group, 'param':param, 'behav':behav, 'param_diff':param_diff, 'behav_diff':behav_diff})
        

    rr,pr = scipy.stats.pearsonr(responders['behav']['both'], responders['param']['both'])

    sbn.regplot(data=pd.DataFrame(lines), x='param_diff', y='behav_diff', ax=ax, color='gray', scatter_kws={'s':10})

    plt.title("n={}  r={:.2f}  p={:.3f}".format(len(responders['behav']['both']),rr,pr), fontsize=10)
    plt.gca().spines.top.set_visible(False)
    plt.gca().spines.right.set_visible(False)
    plt.xlabel("$\Delta \, FC$")
    plt.ylabel("$\Delta \, {}$".format(behav.split('_')[0]))
    

    if gs==None:
        plt.tight_layout()
        if args.save_figs:
            fname = 'FC_YBOCS_scatter'+today()+'.svg'
            plt.savefig(os.path.join(proj_dir, 'img', fname))

        plt.show()



def plot_restoration_figure_paper(df_restore, df_top, df_params_contribution, args):
    """ Concatenates restoration plots for paper """ 
    fig = plt.figure(figsize=[12,10])
    gs = plt.GridSpec(nrows=12, ncols=14)
    plt.tight_layout()

    # top 5 efficacy by targets 
    plot_distance_restore(df_restore, args, gs[:9, 3:9])
    
    # efficacy by number of targets + log scaling
    sub_gs = gs[10:,:10].subgridspec(nrows=1, ncols=5)
    plot_efficacy_by_number_of_target(df_top, sub_gs, args=args)

    # parameters' contribution
    params = args.params
    sub_gs = gs[:9,10:].subgridspec(nrows=3, ncols=2)
    plot_parameters_contribution(df_params_contribution, params, sub_gs, args)

    if args.save_figs:
        fname = 'restoration_combined_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))
    plt.show()


#----------------------#
# IMPROVEMENT ANALYSIS #
#----------------------#

def get_param_zscore(df, params, kdes):
    """ score each parameter based on kdes distributions """
    score = []
    for param in params:
        mean, std = kdes['patients'][param]['vals'].mean(), kdes['patients'][param]['vals'].std()
        zpar = (df[param].iloc[0] - mean) / std
        score.append(zpar)
    return np.array(score)

def score_improvement(df, params, kdes, behav='YBOCS_Total'):
    """ Score parameters based on improvement they induce in functional connectivity (FC) space across virtual interventions. 
    
    Parameters
    ----------
        df: pandas.DataFrame
            Summary data (empirical data paired with digital twins)
        params: list
            Individual intervention targets (i.e. model parameters).
        kdes: dict
            Kernel Density Estimates of posterior distributions of OCD subjects and healthy controls.
        behav: string
            Behavioral measure. Default: Y-BOCS score.
        
    Results
    -------
        df_improvement: pandas.DataFrame
            Normalized differences between initial (pre) and follow-up (post) parameters of digital twins
            for each number of targets in virtual interventions. 
    
    """
    lines = []
    for subj in df.subj.unique():
        df_subj = df[df.subj==subj]

        pre = get_param_zscore(df_subj[df_subj.ses=='ses-pre'], params, kdes)
        post = get_param_zscore(df_subj[df_subj.ses=='ses-post'], params, kdes)
        diff_params = post-pre
        
        pre = df_subj[df_subj.ses=='ses-pre'][behav]
        post = df_subj[df_subj.ses=='ses-post'][behav]
        diff_behav = np.array(pre) - np.array(post) # we want behavioral improvements to be postive

        pre = df_subj[df_subj.ses=='ses-pre']['dist']
        post = df_subj[df_subj.ses=='ses-post']['dist']
        diff_fc = pre.iloc[0] - post.iloc[0]   # we want functional improvements to be positive

        score = diff_params
        score_dict = dict((par,val) for par,val in zip(params,score))
        score_dict['subj'] = subj
        score_dict['group'] = df_subj.group.unique()[0]
        score_dict['diff_behav'] = diff_behav[0]
        score_dict['diff_fc'] = diff_fc
        for param in params:
            pre = df_subj[df_subj.ses=='ses-pre'][param].iloc[0]
            post = df_subj[df_subj.ses=='ses-post'][param].iloc[0]
            score_dict['diff_'+param] = post - pre
        #[score_dict['behav_'+par] = np.abs(diff_behav*val) for par,val in zip(params,score)]
        lines.append(score_dict)
    df_improvement = pd.DataFrame(lines)
    return df_improvement


def plot_improvement_windrose(df_improvement, params, gs=None, args=None):
    """ Make windrose vizualisation of mean improvement in parameter space.
    
    Parameters
    ----------
        df_improvement: pandas.DataFrame
            Z-score normalized differences between initial (pre) and follow-up (post) parameters of digital twins
            for number of targets in virtual interventions. 
        params: list
            Individual intervention targets (i.e. model parameters).
        gs: matplotlib.Gridspec
            (optional) A GridSpec object that can be used to embbed axes when this figure is a subplot 
            of a larger figure.  
        args: argparse.Namespace
            Extra arguments with options. 
    
    """
    theta = np.arange(len(params))/(len(params)) * 2*np.pi
    theta = np.append(theta, theta[0])

    palette = {'group1': 'orange', 'group2': 'green'}

    if gs==None:
        fig, ax = plt.subplots(1,1, subplot_kw={'projection': 'polar'}, figsize=[3,3])
    else:
        ax = plt.subplot(gs, projection='polar')
    
    #for i,group in enumerate(df_improvement.group.unique()):
    sub_df = df_improvement#[df_improvement.group==group]

    line = dict()
    for param in params:
        df_tmp = df_improvement#[df_improvement['diff_behav']>0]
        #line[param], p = scipy.stats.spearmanr(np.array(df_tmp['diff_'+param]), np.array(df_tmp['diff_behav']))
        #print("{:15}  R={:.2f}  p={:.3f}".format(param, line[param], p))
        #line[param] = np.mean(df_tmp[param])
        line[param]= np.correlate(np.array(df_tmp['diff_'+param]), np.array(df_tmp['diff_behav']))[0]
    sub_df = pd.DataFrame([line])    


    # mean improvemet across subjects
    r = np.array(sub_df[params].sum(axis=0))/len(sub_df)
    r = np.append(r, r[0]) 

    #axes[0,i].plot(theta, r, label = str(df_dt.iloc[i].n_test_params), color=palette[df_dt.iloc[i].n_test_params], lw=5)
    #ax.bar(theta, r, label=group, color=palette[group], lw=5, width=0.5, alpha=0.3)
    for theta_, r_ in zip(theta, r):
        if r_ > 0:
            # increase of parameter due to treatment 
            #ax.bar(theta_, np.abs(r_), color='red', lw=5, width=0.5, alpha=0.2+np.abs(r_)/2)
            ax.bar(theta_, np.abs(r_), color='goldenrod', width=0.5, alpha=0.2+np.abs(r_)/100, lw=1, edgecolor='k', linestyle='-')
        else:
            # decreae of paramter due to treatment
            #ax.bar(theta_, np.abs(r_), color='blue', lw=5, width=0.5, alpha=0.2+np.abs(r_)/2)
            ax.bar(theta_, np.abs(r_), color='goldenrod', width=0.5, alpha=0.2+np.abs(r_)/100, lw=1, edgecolor='k', linestyle='--')

    #axes[i].fill_between(theta, rmin, rmax)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(params)
    lbls = ax.get_xticklabels()
    new_lbls = format_labels(lbls)
    ax.set_xticklabels(new_lbls)
    #axes[i].set_rticks([0.1], labels=[])
    ax.set_yticklabels([])
    ax.set_yticks([1])
    #ax.set_rmin(-1)
    #ax.set_rmax(1)
    ax.spines.polar.set_visible(False)
    ax.xaxis.grid(linewidth=0.5, linestyle='--')

    if gs==None:
        plt.tight_layout()
        if args.save_figs:
            #plt.rcParams['svg.fonttype'] = 'none'
            fname = 'improvement_params_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
            plt.savefig(os.path.join(proj_dir, 'img', fname))
        plt.show()


def plot_improvement_bars(df_improvement, params, gs=None, args=None):
    """ Make barplot vizualisation of improvements in parameter space.
    
    Parameters
    ----------
        df_improvement: pandas.DataFrame
            Z-score normalized differences between initial (pre) and follow-up (post) parameters of digital twins
            for number of targets in virtual interventions. 
        params: list
            Individual intervention targets (i.e. model parameters).
        gs: matplotlib.Gridspec
            (optional) A GridSpec object that can be used to embbed axes when this figure is a subplot 
            of a larger figure.  
        args: argparse.Namespace
            Extra arguments with options. 
    
    """

    if gs==None:
        fig, ax = plt.subplots(1,1, figsize=[3,6])
    else:
        ax = plt.subplot(gs)
    
    # 1) behavioral improvement during treatment
    sub_df = df_improvement[sub_df.diff_behav>0] #[df_improvement.group==group]

    for i,param in enumerate(params):
        mean_param_diff = sub_df[param].mean()
        # increased params 
        if mean_param_diff>0:
            ax.bar(i, np.abs(mean_param_diff)*sub_df['diff_behav'], color='goldenrod', width=0.5, alpha=0.3, lw=1, edgecolor='k', linestyle='-')
        # behavioral recession during treatment
        else:
            ax.bar(i, np.abs(mean_param_diff)*sub_df['diff_behav'], color='goldenrod', width=0.5, alpha=0.3, lw=1, edgecolor='k', linestyle='-')

    #axes[i].fill_between(theta, rmin, rmax)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(params)
    lbls = ax.get_xticklabels()
    new_lbls = format_labels(lbls)
    ax.set_xticklabels(new_lbls)
    #axes[i].set_rticks([0.1], labels=[])
    ax.set_yticklabels([])
    ax.set_yticks([1])
    #ax.set_rmin(-1)
    #ax.set_rmax(1)
    ax.spines.polar.set_visible(False)
    ax.xaxis.grid(linewidth=0.5, linestyle='--')

    if gs==None:
        plt.tight_layout()
        if args.save_figs:
            #plt.rcParams['svg.fonttype'] = 'none'
            fname = 'improvement_params_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
            plt.savefig(os.path.join(proj_dir, 'img', fname))
        plt.show()


def plot_improvement_distrib(df_improvement, param_imp=None):
    plt.figure(figsize=[12,3])
    df_imp = df_improvement.melt(id_vars=['subj', 'group'], value_vars=params, var_name='param', value_name='diff', ignore_index=True)
    sbn.swarmplot(data=df_imp, x='param', y='diff', hue='group', dodge=True, palette={'group1':'orange', 'group2':'green'}, alpha=0.5, size=3)
    plt.hlines(0, -1, len(params), color='black', linestyle='--', lw=0.5, alpha=0.5)
    plt.legend().set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.gca().spines.right.set_visible(False)
    if param_imp==None:
        plt.title('Normalized improvement')
    else:
        plt.title('Normalized improvement scaled by '+param_imp)
    plt.show()


def print_stats_improvement(df_improvement, params, param_imp=None):
    print("STATS FOR NORMALIZED PARAMETER IMPROVEMENT SCALED BY "+str(param_imp))
    for param in params:
        gp1 = df_improvement[df_improvement.group=='group1'][param]
        gp2 = df_improvement[df_improvement.group=='group2'][param]
        t,p = scipy.stats.ttest_ind(gp1, gp2)
        print("{:12} \t t={:.2f} \t p={:.3f}".format(param, t, p))


def plot_improvement_pre_post_params(df_summary, params, args):
    """ Plot initial (pre) and follow-up (post) distributions of parameters from digital twin analysis. 
    
    Parameters
    ----------
        df_summary: pandas.DataFrame
            Summary measures of empirical analysis, inluding subject's functional connectivity across frontostriatal circuits,
            behavioral measures (e.g. Y-BOCS score, IQ, etc.), digital pairing and values of the digital twin parameters.
        params: list
            Individual intervention targets (i.e. model parameters).
        args: argparse.Namespace
            Extra arguments with options. 
    """
    plt.figure(figsize=[22,3])
    for i,param in enumerate(params):
        plt.subplot(1, len(params), i+1)
        df_sum = df_summary.melt(id_vars=['subj', 'ses'], value_vars=param, var_name='param', ignore_index=True)
        sbn.swarmplot(data=df_sum, x='param', y='value', hue='ses', hue_order=['ses-pre', 'ses-post'], dodge=True, palette={'ses-pre':'orange', 'ses-post':'purple'}, size=3.5, alpha=0.5)
        plt.legend().set_visible(False)
        plt.gca().spines.top.set_visible(False)
        plt.gca().spines.right.set_visible(False)
        #plt.gca().spines.bottom.set_visible(False)
        s_pre, p_pre = scipy.stats.normaltest(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value)
        s_post, p_post = scipy.stats.normaltest(df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        u,p = scipy.stats.mannwhitneyu(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        d = cohen_d(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        t,pt = scipy.stats.ttest_rel(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        T,pT = scipy.stats.wilcoxon(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        KS,pKS = scipy.stats.ks_2samp(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        ES,pES = scipy.stats.epps_singleton_2samp(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        print("{:15}: normality={:1}/{:1}   d={: 1.2}    u={:5}  p={:.4f}    t={: .2f}  p={:.4f}   T={}  p={:.4f}  KS={:.3f}  p={:.4f}  ES={:.3f}  p={:.4f}".format(param, p_pre>0.05, p_post>0.05,d, int(u), p, t, pt,int(T), pT, KS, pKS, ES, pES))
        #plt.title("u={}  p={:.3f}".format(int(u),p))
        ttl = ''
        #if ((p_pre>0.05) and (p_post>0.05)):
        #    if p<.05:
        #        ttl = ttl+'*'
        #    if p*len(params)<0.05:
        #        ttl = ttl+'*'
        #else:
        if pT<.05:
            ttl = ttl+'$\star$'
        if pT*len(params)<0.05:
            ttl = ttl+'$\star$'
        plt.title(ttl, fontsize=14)
        #plt.ylabel("${}$".format(format_param(param)), fontsize=12)
        plt.xticks([0], labels=[])
        plt.xlabel("${}$".format(format_param(param)), fontsize=12)
        plt.ylabel('')

        if param.startswith('C_'):
            plt.ylim([-0.12, 0.55])
            if i>1:
                plt.gca().spines.left.set_visible(False)
                plt.yticks([], labels=None)
        elif param.startswith('eta_'):
            plt.ylim([-0.01, 0.11])

        if args.save_figs:
            fname = 'distrib_params_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
            plt.savefig(os.path.join(proj_dir, 'img', fname))

    plt.tight_layout()
    plt.show()


def get_kde(data, mn, mx, smoothing_factor=10):
    """ create kernel density estimate for the data (used in violin-like plots) """
    b = (mx-mn)/smoothing_factor
    mn -= (mx-mn)*0.1
    mx += (mx-mn)*0.1
    model = sklearn.neighbors.KernelDensity(bandwidth=b)
    xtrain = np.array(data)[:, np.newaxis]
    model.fit(xtrain)
    xtest = np.linspace(mn,mx,100)[:, np.newaxis]
    log_dens = model.score_samples(xtest)
    mu = model.score_samples(np.mean(data).reshape(1,1))
    return xtest, np.exp(log_dens), np.exp(mu)


def plot_improvement_pre_post_params_paper(df_summary, params, gs=None, args=None):
    """ Plot initial (pre) and follow-up (post) distributions of parameters from digital twin analysis. 
    (only relevant parameters are shown for manuscript). 
    
    Parameters
    ----------
        df_summary: pandas.DataFrame
            Summary measures of empirical analysis, inluding subject's functional connectivity across frontostriatal circuits,
            behavioral measures (e.g. Y-BOCS score, IQ, etc.), digital pairing and values of the digital twin parameters.
        params: list
            Individual intervention targets (i.e. model parameters).
        gs: matplotlib.Gridspec
            (optional) A GridSpec object that can be used to embbed axes when this figure is a subplot 
            of a larger figure.
        args: argparse.Namespace
            Extra arguments with options. 
    """
    #i_params = [1,2,5,7] # indices of params to plot
    i_params = [1,3,5] # indices of params to plot
    #i_params = [1,2,3,5] # indices of params to plot
    #i_params = [1,3,5,8] # indices of params to plot
    

    if gs==None:
        fig = plt.figure(figsize=[6,2])
        sub_gs = plt.GridSpec(nrows=1, ncols=len(i_params)+1)
    else:
        #sub_gs = gs.subgridspec(nrows=1, ncols=len(i_params)+1, width_ratios=[1,1,1,0.5,1])
        sub_gs = gs.subgridspec(nrows=1, ncols=len(i_params))#, width_ratios=[1,1,1,0.5,1])

    for i,i_param in enumerate(i_params):
        param = params[i_param]
        #plt.subplot(1, 4, i+1)
        if (param.startswith('eta_') | param.startswith('sigma')):
            ax = plt.subplot(sub_gs[0,i+1])
        else:
            ax = plt.subplot(sub_gs[0,i])
        df_sum = df_summary.melt(id_vars=['subj', 'ses'], value_vars=param, var_name='param', ignore_index=True)

        #sbn.swarmplot(data=df_sum, x='param', y='value', hue='ses', hue_order=['ses-pre', 'ses-post'], dodge=True, 
        #              palette={'ses-pre':'orange', 'ses-post':'purple'}, size=3, alpha=0.5, ax=ax)
        for subj in df_sum.subj.unique():
            sbn.pointplot(data = df_sum[df_sum.subj==subj], y='value', x='ses', order=['ses-pre', 'ses-post'], dodge=True, 
                          color='gray', linewidth=0.5, marker={'size':1}, size=0.5, alpha=0.5, ax=ax)

        kde_scale = 0.1    
        kde_pre_x, kde_pre_logdensity, kde_pre_exp_mu = get_kde(np.array(df_sum[df_sum.ses=='ses-pre'].value), 
                                                                mn=df_sum.value.min(), mx=df_sum.value.max())
        kde_post_x, kde_post_logdensity, kde_post_exp_mu = get_kde(np.array(df_sum[df_sum.ses=='ses-post'].value),
                                                                   mn=df_sum.value.min(), mx=df_sum.value.max())
        
        ax.fill(-kde_pre_logdensity/kde_pre_logdensity.std()*kde_scale,kde_pre_x, color='goldenrod', alpha=0.2)
        #ax.plot([-kde_pre_exp_mu/kde_pre_logdensity.std()*kde_scale,0],[df_sum[df_sum.ses=='ses-pre'].value.mean(), df_sum[df_sum.ses=='ses-pre'].value.mean()],
        #         '-', color='gray')
        
        ax.fill(1+kde_post_logdensity/kde_post_logdensity.std()*kde_scale,kde_post_x, color='goldenrod', alpha=0.2)
        #ax.plot([1+kde_post_exp_mu/kde_post_logdensity.std()*kde_scale,0],[df_sum[df_sum.ses=='ses-post'].value.mean(), df_sum[df_sum.ses=='ses-post'].value.mean()],
        #         '-', color='gray')


        plt.setp(ax.collections, sizes=[6], linewidth=[0.1])
        plt.setp(ax.lines, linewidth=0.3)
        plt.legend().set_visible(False)
        plt.gca().spines.top.set_visible(False)
        plt.gca().spines.right.set_visible(False)
        #plt.gca().spines.bottom.set_visible(False)
        u,p = scipy.stats.mannwhitneyu(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        d = cohen_d(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        t,pt = scipy.stats.ttest_rel(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        T,pT = scipy.stats.wilcoxon(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        ES,pES = scipy.stats.epps_singleton_2samp(df_sum[df_sum.ses=='ses-pre'].sort_values('subj').value, df_sum[df_sum.ses=='ses-post'].sort_values('subj').value)
        print("{:15}: d={: 1.2}    u={:5}  p={:.4f}    t={: .2f}  p={:.4f}   T={: }  p={:.4f}  ES={:.3f}  pES={:.4f}".format(param, d, int(u), p, t, pt,int(T), pT, ES, pES))
        #plt.title("u={}  p={:.3f}".format(int(u),p))
        ttl = ''
        ft = 10
        if pT<.05:
            ttl = ttl+'*'
            if pT*len(i_params)<0.05:
                ttl = ttl+'*'
            ft=14
        #if pES<0.05:
        #    ttl += '#'
        #    if pES*len(i_params)<0.05:
        #        ttl += '#'
        ttl = plt.title(ttl, fontsize=ft)
        x,y = ttl.get_position()
        ttl.set_position([x,y-0.2])
        #plt.ylabel("${}$".format(format_param(param)), fontsize=12)
        plt.xticks([0,1], labels=['baseline', 'post'])
        plt.xlabel("${}$".format(format_param(param)), fontsize=12)
        if i==0:
            plt.ylabel('value')
        else:
            plt.ylabel('')

        if param.startswith('C_'):
            #ax.set_ylim([-0.12, 0.55])
            ax.set_ylim([-0.17, 0.55])
            if i>0:
                plt.gca().spines.left.set_visible(False)
                plt.yticks([], labels=None)
                plt.subplots_adjust(left=0, right=0.8)
        elif param.startswith('eta_'):
            ax.set_ylim([-0.01, 0.11])
            #ax.set_ylim([-0.03, 0.13])
            plt.yticks([0, 0.05, 0.1])
            plt.subplots_adjust(left=0.2, right=1)
        elif param.startswith('sigma'):
            ax.set_ylim([0.05, 0.1])
            #ax.set_ylim([-0.03, 0.13])
            plt.yticks([0.06, 0.08])
            plt.subplots_adjust(left=0.2, right=1)
    plt.tight_layout()
    
    if gs==None:
        plt.tight_layout()
        if args.save_figs:
                fname = 'distrib_params_paper_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg'
                plt.savefig(os.path.join(proj_dir, 'img', fname))
        plt.show()


def drop_unimproved(df_summary, df_improvement, feature='dist', threshold=0.1):
    """ Remove subjects whos feature do not improve between pre and post. 
    
    Parameters
    ----------
        df_summary: pandas.DataFrame 
            Summary output of the analysis
        df_improvement: pandas.DataFrame
            Summary improvement of the virtual interventions
        feature: string
            Column of the dataframe to apply filtering on.

    Returns
    -------
        df_summary: pandas.DataFrame 
            Summary output of the analysis
        df_improvement: pandas.DataFrame
            Summary improvement of the virtual interventions
            
    """ 
    for subj in df_summary.subj.unique():
        df = df_summary[df_summary.subj==subj]
        if (df[df.ses=='ses-post'][feature].iloc[0] - df[df.ses=='ses-pre'][feature].iloc[0] > threshold):
            df_summary.drop(df_summary[df_summary.subj==subj].index[:], inplace=True)
            df_improvement.drop(df_improvement[df_improvement.subj==subj].index[:], inplace=True)
    return df_summary, df_improvement


def plot_summary_improvement(df_summary, df_improvement, params, feature=None, args=None):
    """ Just a layout organization to have all plots in same figure """
    fig = plt.figure(figsize=[6,5])
    gs = plt.GridSpec(nrows=2,ncols=4, height_ratios=[1,1])

    #df_summary, df_improvement = drop_unimproved(df_summary, df_improvement, threshold=args.fc_thresh)

    # FC - YBOCS scatter + regrression plot
    plot_pre_post_dist_ybocs(df_summary, gs=gs[0,0:2], args=args)

    # Param contribution 
    plot_improvement_windrose(df_improvement, params=params, gs=gs[0,2:4], args=args)

    # pre-post change in param space
    plot_improvement_pre_post_params_paper(df_summary, params=params, gs=gs[1,:], args=args)

    plt.tight_layout()

    #plot_improvement_pre_post_params(df_summary, params=params, args=args)

    if args.save_figs:
        fname = os.path.join(proj_dir, 'img', 'summary_improvement_'+args.distance_metric+'_'+args.efficacy_base+today()+'.svg')
        plt.savefig(fname)
    
    plt.show()


def linear_regression_sims(df_restore, params):
    """ Linear model between change in parameters and change in distance to healthy controls """ 
    df_rest = df_restore[(df_restore.ustat>96200)]# & (df_restore.dist<df_restore.dist_pre_hc)]
    #df_rest = df_restore
    #df_rest = df_rest.reset_index()

    X = []
    inds = dict((i,[]) for i in np.arange(0,7))
    y = []
    for j,row in df_rest.iterrows():
        x = np.zeros(len(params),)
        for i,par in enumerate(params):
            #if par in tp.split(' '):
            x[i] = row['z_'+par]
        X.append(x)
        inds[row.n_test_params].append(j)
        #y.append(row['ustat'])
        y.append(row['dist_pre_hc'] - row['dist'])

    X = np.array(X)
    y = np.array(y)

    model = sklearn.linear_model.LinearRegression(positive=True)
    model.fit(X,y)

    score = model.score(X,y)

    return model, score


def plot_regression_coefs(model, score, params):
    plt.bar(np.arange(1,12), model.coef_)
    plt.xticks(np.arange(1,12), labels=params, rotation=60)
    plt.title('$R^2={:.2f}$'.format(score))


def linear_regression_digital_twins(df_improvement, params):
    """ Linear model between change in parameters and improvement in YBOCS """ 
    df_imp = df_improvement[df_improvement['diff_behav']>0]
    #df_imp = df_improvement#[df_improvement['diff_fc']>0]
    X = np.array(df_imp[params])
    y = np.array(df_imp['diff_behav'])
    #y = np.array(df_imp['diff_fc'])

    model = sklearn.linear_model.LinearRegression(positive=True)
    mdoel.fit(X,y)

    score = model.score(X,y)

    return model


def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()

    # global parameters
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--save_summary', default=False, action='store_true', help='save output summary')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--n_batch', type=int, default=10, action='store', help="batch size")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    
    # loading parameters
    parser.add_argument('--db_names', type=str, nargs='+', default=["sim_pat_20230628", "sim_pat_20230721"], help="identifier of the sqlite3 database (use sim_pat_20230628 for digital twin, sim_pat_20230721 for restoration analysis")
    parser.add_argument('--load_sims', default=False, action='store_true', help='Load simulations based on db_names')

    # digital twin  
    parser.add_argument('--tolerance', type=float, default=0.05, action='store', help="maximal distance allow to take into consideration 'digital sigling'")
    parser.add_argument('--tolerance_plot', type=float, default=0.3, action='store', help="maximal distance allow to take into consideration 'digital sigling' (for plotting) ")
    parser.add_argument('--save_distances', default=False, action='store_true', help='save distances between patients and simulations')
    parser.add_argument('--load_distances', default=False, action='store_true', help='load distances between patients and simulations from previously saved')
    parser.add_argument('--compute_distances', default=False, action='store_true', help='compute distances between patients and simulations')
    parser.add_argument('--compute_post_distances', default=False, action='store_true', help='Compute distances for digital twins analysis using post-TMS FC')
    parser.add_argument('--n_closest', type=int, default=1, action='store', help="number of digital twins to retain sorted by increasing distance")
    parser.add_argument('--plot_param_behav', default=False, action='store_true', help='plot param-behavioral relationship')
    parser.add_argument('--verbose', default=False, action='store_true', help='print extra processing info')
    parser.add_argument('--session', default=None, action='store', help='which session (ses-pre or ses-post) for behavioral scores (default:None => both are used')
    
    # multivariate analysis (not reported in paper)
    parser.add_argument('--multivariate_analysis', default=False, action='store_true', help='perform multivariate analysis on simulations parameters')
    parser.add_argument('--multivar_fc', default=False, action='store_true', help='perform multivariate analysis on FC variables')
    parser.add_argument('--cv_type', type=str, default='LeaveOneOut', help="which cross-validation scheme to apply (KFold, RepeatedKFold, ShuffleSplit, LeaveOneOut)")
    parser.add_argument('--n_splits', type=int, default=10, help="number of splits used for cross-validation")
    parser.add_argument('--n_repeats', type=int, default=5, help="number of repetitions for the RepeatedKFold cross-validation")
    parser.add_argument('--max_depth', type=int, default=3, action='store', help="max depth of the decision tree")
    parser.add_argument('--test_size', type=float, default=0.3, help="ratio of test data (over all data) used for ShuffleSplit cross-validation")
    parser.add_argument('--plot_cv_regression', default=False, action='store_true', help='plot cross validation regression scatters')
    parser.add_argument('--plot_multivar_svd', default=False, action='store_true', help='plot dimensionality reduction on regression coefficients')
    parser.add_argument('--plot_multivariate_results', default=False, action='store_true', help='plot multivariate linear regression coefficients')
    
    parser.add_argument('--null', default=False, action='store_true', help='shuffle coupling weights to create null hypothesis for regression coefficients')
    parser.add_argument('--n_null', type=int, default=100, action='store', help="number of elements to make null distribution")
    parser.add_argument('--plot_null_distrib', default=False, action='store_true', help='plot null distribution analysis of linear regression coefficients')
    parser.add_argument('--plot_cv_null_distrib', default=False, action='store_true', help='plot null distribution analysis of linear regression coefficients for all CV folds')
    
    parser.add_argument('--print_ANOVA', default=False, action='store_true', help='print stats for mixed and multiple one-way ANOVAs')
    
    # restoration analysis parameters
    parser.add_argument('--restore_analysis', default=False, action='store_true', help='perform retoration analys of test parameters to move from patient to controls FC')
    parser.add_argument('--base_cohort', type=str, default='controls', help="Cohort from which to infer posterior as default")
    parser.add_argument('--test_cohort', type=str, default='patients', help="Cohort from which to infer posterior of individual params")
    parser.add_argument('--plot_efficacy_transform', default=False, action='store_true', help='plot the transformation from distance from controls to efficacy')
    parser.add_argument('--n_restore', type=int, default=10, action='store', help="number of best restorations for plotting")
    parser.add_argument('--n_tops', type=int, default=5, action='store', help="number of best restorations for each n_test_param for plotting")
    parser.add_argument('--n_test_params', type=int, default=6, action='store', help="max number of parameter combinations for plotting restoration outputs")
    parser.add_argument('--distance_metric', type=str, default='rmse', help="distance used in restoration metric (rmse or emd)")
    parser.add_argument('--efficacy_base', type=str, default='sims', help="use simulated centroid (sims) or paired (paired) or data (anything else) group difference in efficacy denominator")
    parser.add_argument('--use_optim_params', default=False, action='store_true', help='flag if using best optimization outputs directly (default: No, i.e. draw new params from posteriors.')
    parser.add_argument('--sort_style', type=str, default='by_n', help="how to sort distances for visualization: 'by_n' or 'all' (default)")
    parser.add_argument('--save_restoration', default=False, action='store_true', help='save outputs of the restoration analysis (df_restore)')
    parser.add_argument('--load_restoration', default=False, action='store_true', help='load outputs from previously saved restoration analysis (df_restore)')
    parser.add_argument('--plot_distance_restore', default=False, action='store_true', help='plot efficacy horizontal box plots')
    parser.add_argument('--plot_restoration_figure_paper', default=False, action='store_true', help='plot figure for paper with layout containing sub-figures')
    
    parser.add_argument('--contribution_analysis', default=False, action='store_true', help='run parameter contribution analysis')
    parser.add_argument('--sensitivity_analysis', default=False, action='store_true', help='run paramater sensitivity analysis')
    parser.add_argument('--load_param_contribution', default=False, action='store_true', help='load parameter contributions from previously computed restoration analysis')
    parser.add_argument('--load_param_sensitivity', default=False, action='store_true', help='load parameter sensitivity from previously computed restoration analysis')
    parser.add_argument('--plot_params_contribution_sensitivity', default=False, action='store_true', help='plot contribution and sensitivity of parameters in restoration analysis')

    parser.add_argument('--predictive_analysis', default=False, action='store_true', help='Analyse predictive power of model based on distance to controls FC')
    parser.add_argument('--plot_fc_dist_pre_post_behav', default=False, action='store_true', help='plot FC distance to controls in pre-post TMS data')
    parser.add_argument('--compute_sim_vecs', default=False, action='store_true', help='Compute simulation vectors in FC space (otherwise load pre-computed)')
    parser.add_argument('--save_sim_vecs', default=False, action='store_true', help='Save simulation vectors in FC space after being computed')
    parser.add_argument('--load_sim_vecs', default=False, action='store_true', help='Load simulation vectors in FC space')
    parser.add_argument('--load_post_distances', default=False, action='store_true', help='load distances in post-TMS inference')
    parser.add_argument('--plot_pre_post_associations', default=False, action='store_true', help='plot pre-post TMS associations between FC, params and behavioral measures')

    parser.add_argument('--load_summary', default=False, action='store_true', help='load summary of digital twin analysis')
    parser.add_argument('--improvement_analysis', default=False, action='store_true', help='Analyse improvement of models parameters pre-post TMS')
    parser.add_argument('--param_improvement', type=str, default=None, help="defines which parameter to scale improvement with")
    parser.add_argument('--fc_thresh', type=float, default=0, action='store', help="fc-ybocs reg plot -- theshold for fc improvement")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    # load histories and KDEs
    #histories = import_results(args)
    
    behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total', 'Dep_Total', 'Anx_total']
    params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42', 'eta_C_13', 'eta_C_24', 'sigma', 'sigma_C_13', 'sigma_C_24']

    if args.load_sims:
        print("Loading simulations...")
        df_sims = load_df_sims(args)
        
        print("Fixing simulations indices...")
        fix_df_sims_names(df_sims, args)
        
    print("Loading functional connectivity data...")
    df_data = load_df_data(args)
    pathways = np.sort(df_data.pathway.unique())
    
    print("Loading behavioral data...")
    with open(os.path.join(proj_dir, 'postprocessing', 'df_pat_.pkl'), 'rb') as f:
        df_pat = pickle.load(f)
        if args.session!=None:
            df_pat = df_pat[df_pat.ses==args.session]
    df_fc_pat = df_data.pivot(columns='pathway', values='corr', index='subj').reset_index().merge(df_pat)

    kdes = load_kdes(args)

    if args.compute_sim_vecs:
        print("Computing simulation vectors...")
        sim_vecs = compute_sim_vecs(df_sims)
        if args.save_sim_vecs:
            print("Saving simulation vectors...")
            fname = os.path.join(proj_dir, 'postprocessing', 'sim_vecs'+today()+'.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(sim_vecs, f)
    elif args.load_sim_vecs:
        print("Loading simulation vectors...")
        fname = os.path.join(proj_dir, 'postprocessing', 'sim_vecs.pkl') #sim_vecs.pkl without restoration, sim_vecs_20230915.pkl with restoration
        with open(fname, 'rb') as f:
            sim_vecs = pickle.load(f)

    
    if args.load_distances:
        print("Loading distances...")
        #fname = os.path.join(proj_dir, 'postprocessing', args.db_names[0]+'_distances100eps'+str(int(args.tolerance*100))+".pkl")
        #fname = os.path.join(proj_dir, 'postprocessing', 'assoc_distances100eps'+str(int(args.tolerance*100))+"_pre_20230918.pkl")
        fname = os.path.join(proj_dir, 'postprocessing', 'assoc_digital_twins_distances100eps'+str(int(args.tolerance*100))+"_pre_20240329.pkl")
        with open(fname, 'rb') as f:
            assoc = pickle.load(f)
        df_sim_pre = merge_data_sim_dfs(df_pat[df_pat.ses=='ses-pre'], df_sims, assoc, args)
    elif args.compute_distances:
        print("Computing distances...")
        assoc = compute_distances(df_data, df_sims[df_sims.test_param=='None'], ses='pre', args=args)
        df_sim_pre = merge_data_sim_dfs(df_pat[df_pat.ses=='ses-pre'], df_sims, assoc, args)
    

    # univariate analysis
    if args.plot_param_behav:
        plot_param_behav(df_sim_pre, params=params, args=args)

    # multivariate analysis
    if args.multivariate_analysis:
        print("Running multivariate analysis...")
        if args.multivar_fc:
            params=['Acc_OFC', 'Acc_PFC', 'Acc_dPut', 'OFC_PFC', 'dPut_PFC']
            multivar = multivariate_analysis(df_fc_pat, 
                                             params=params, 
                                             args=args)
        else:
            multivar = multivariate_analysis(df_sim_pre, params=params, args=args)

        if args.plot_multivariate_results:
            plot_multivariate_results(multivar, args=args)
        
        if args.plot_cv_regression:
            plot_cv_regression(multivar, df_sim_pre, args=args)
        
        if args.plot_multivar_svd:
            plot_multivar_svd(multivar, behavs=behavs, args=args)

        if args.null:
            print("Creating Null distribution...")
            models = dict(('null_model{:04d}'.format(i), sklearn.linear_model.Ridge(alpha=0.01)) 
                                for i in range(args.n_null))
            multivar_null = multivariate_analysis(df_sim_pre, 
                                                  params=params, 
                                                  models=models, 
                                                  null=True, 
                                                  args=args)

            create_df_null(multivar, multivar_null)

            if args.plot_null_distrib:
                plot_null_distrib(multivar, args)

        if args.print_ANOVA:
            print_ANOVA(df_sim_pre, behavs, params)


    # restoration analysis
    if args.restore_analysis:
        print("Restoration analysis...")
        args.pathways = pathways
        args.params = params

        if args.plot_efficacy_transform:
            plot_efficacy_transform(args)

        if args.load_restoration:
            optim=''
            date = '_20240326'
            if args.use_optim_params:
                optim += '_optim'
                date = '_20240329'
                
            if 'paired' in args.efficacy_base:
                #restoration_file = 'df_restore_'+args.distance_metric+'_paired_20240309.pkl'
                restoration_file = 'df_restore_'+args.distance_metric+optim+'_'+args.efficacy_base+date+'.pkl'
            else:
                restoration_file = 'df_restore_'+args.distance_metric+'_20240313.pkl'
            
            with open(os.path.join(proj_dir, 'postprocessing', restoration_file), 'rb') as f:
                df_restore = pickle.load(f)
        else:
            df_restore = compute_distance_restore(df_sims[df_sims.test_param!='None'], args)
            
            if args.save_restoration:
                fname = 'df_restore'+ get_restoration_suffix(args) + today() + '.pkl'
                with open(os.path.join(proj_dir, 'postprocessing', fname), 'wb') as f:
                    pickle.dump(df_restore, f)
            
        df_restore = compute_efficacy(df_restore, args=args)

        if args.plot_distance_restore:
            plot_distance_restore(df_restore, args=args)

        df_top, top_params = get_df_top(df_restore, args)
        
        #df_feature_importance, decision_trees, feat_imps = decision_tree(df_restore, params, args)
        #df_custom_feat_imps = compute_custom_feature_scores(decision_trees, params, args)
        #df_simple_feat_imps = compute_simple_feature_scores(df_top, params, args)

        #df_reliability = compute_feature_reliability(df_top, params, kdes, args)
        #plot_contribution_windrose(df_reliability, params=['z_'+param for param in params], args=args)

        #df_scaled_efficacy = compute_scaled_feature_score(df_restore[df_restore.efficacy>0], params, kdes, scaling='contribution', args=args)
        #plot_contribution_windrose(df_scaled_efficacy, params, args=args)

        # Parameters contribution
        if args.contribution_analysis:
            if args.load_param_contribution:
                with open(os.path.join(proj_dir, 'postprocessing', 'df_param_contribution.pkl'), 'rb') as f:
                    df_params_contribution = pickle.load(f)
            else:
                df_params_contribution = compute_scaled_feature_score(df_restore[df_restore.efficacy>0.5], params, kdes, scaling='cross_correlation', args=args)
                #df_params_contribution = compute_scaled_feature_score(df_restore, params, kdes, scaling='contribution', args=args)
                if args.save_outputs:
                    with open(os.path.join(proj_dir, 'postprocessing', 'df_param_contribution_'+args.distance_metric+'_'+args.efficacy_base+today()+'.pkl'), 'wb') as f:
                        pickle.dump(df_params_contribution, f)        
            if args.plot_params_contribution_sensitivity:
                plot_contribution_windrose(df_params_contribution, params, args=args)
            
            if args.plot_restoration_figure_paper:
                plot_restoration_figure_paper(df_restore, df_top, df_params_contribution, args)

        # Parameter sensitivity (deprecated alternative to contribution)
        if args.sensitivity_analysis:
            # Parameters sensitivity
            if args.load_param_sensitivity:
                with open(os.path.join(proj_dir, 'postprocessing', 'df_param_sensitivity.pkl'), 'rb') as f:
                    df_param_sensitivity = pickle.load(f)
            else:
                df_param_sensitivity = compute_scaled_feature_score(df_restore[df_restore.efficacy>0], params, kdes, scaling='sensitivity', args=args)
                if args.save_outputs:
                    with open(os.path.join(proj_dir, 'postprocessing', 'df_param_sensitivity_'+args.distance_metric+'_'+args.efficacy_base+today()+'.pkl'), 'wb') as f:
                        pickle.dump(df_param_sensitivity, f)        
            if args.plot_params_contribution_sensitivity:
                plot_sensitivity_windrose(df_param_sensitivity, params, args=args)
        
        


    # prediction 
    if args.predictive_analysis:
        print('Load functional distances (data)...')
        df_fc_pre_post = load_dist_to_FC_controls(args)

        # delta FC vs delta YBOCS
        df_dist_fc = df_fc_pre_post.merge(df_fc_pat, on=['subj', 'ses', 'group', *pathways], how='inner')    
        #df_dist_fc = drop_single_session(df_dist_fc)
        if args.plot_fc_dist_pre_post_behav:
            plot_fc_dist_pre_post_behav(df_dist_fc, args=args)


        # Load or compute distances (post)
        if args.compute_post_distances:
            print("Computing functional distances (post)...")
            pathways = np.sort(df_data.pathway.unique())
            df_post_data = df_fc_pre_post[df_fc_pre_post.ses=='ses-post'].melt(id_vars=['subj', 'cohort', 'group'], 
                                               value_vars=pathways, var_name='pathway', value_name='corr')
            assoc_post = compute_distances(df_post_data, df_sims, ses='post', args=args)

        elif args.load_post_distances:
            print("Loading functional distances (post)...")
            #fname = os.path.join(proj_dir, 'postprocessing', args.db_names[0]+'_distances100eps'+str(int(args.tolerance*100))+"_post.pkl")
            #fname = os.path.join(proj_dir, 'postprocessing', 'assoc_distances100eps'+str(int(args.tolerance*100))+"_post_20230918.pkl")
            fname = os.path.join(proj_dir, 'postprocessing', 'assoc_digital_twins_distances100eps'+str(int(args.tolerance*100))+"_post_20240329.pkl")
            with open(fname, 'rb') as f:
                assoc_post = pickle.load(f) 
        
        df_sim_post = merge_data_sim_dfs(df_pat[df_pat.ses=='ses-post'], df_sims, assoc_post, args)
        df_post = pd.merge(df_fc_pre_post[df_fc_pre_post.ses=='ses-post'], df_sim_post, on=['subj', 'ses', 'group'],
                            how='outer', suffixes=[None, '_sim'])


        # pre-post analysis summary
        df_pre = pd.merge(df_fc_pre_post[df_fc_pre_post.ses=='ses-pre'], 
                          df_sim_pre, 
                          on=['subj', 'ses', 'group'], how='outer', suffixes=[None, '_sim'])
        
        df_summary = df_post[df_post.cohort=='patients'].merge(df_pre[df_pre.cohort=='patients'], how='outer')
        

        if args.save_summary:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_pre_'+args.distance_metric+'_'+args.efficacy_base+today()+".pkl"), 'wb') as f:
                pickle.dump(df_pre, f)
            with open(os.path.join(proj_dir, 'postprocessing', 'df_post_'+args.distance_metric+'_'+args.efficacy_base+today()+".pkl"), 'wb') as f:
                pickle.dump(df_post, f)
            with open(os.path.join(proj_dir, 'postprocessing', 'df_summary_'+args.distance_metric+'_'+args.efficacy_base+today()+".pkl"), 'wb') as f:
                pickle.dump(df_summary, f)
            
    
        if args.plot_pre_post_associations:
            plot_fc_dist_pre_post_behav(df_summary)
            plot_fc_dist_pre_post_params(df_summary)
            #plot_pre_post_params_behavs(df_summary)
    

    if args.load_summary:
        # df_summary_20230915.pkl : pre and post with controls/patients param swaps
        # df_summary_20230918.pkl : only post with param swaps
        with open(os.path.join(proj_dir, 'postprocessing', 'df_summary_20230918.pkl'), 'rb') as f:
                df_summary = pickle.load(f)

    if args.improvement_analysis:
        df_improvement = score_improvement(df_summary, params, kdes)
        plot_summary_improvement(df_summary, df_improvement, params, args=args)
        


