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
import pdb
import pyabc
import scipy
import seaborn as sbn
import sklearn
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit, LeaveOneOut
from sklearn.metrics import make_scorer, r2_score
from sklearn.inspection import permutation_importance
import sqlite3 as sl
import statsmodels
import statsmodels.stats.weightstats

# import most relevant environment and project variable
from OCD_modeling.utils.utils import *
from OCD_modeling.mcmc.history_analysis import import_results, compute_kdes
from OCD_modeling.mcmc.simulate_inference import batched
from OCD_modeling.analysis.fc_data_analysis import drop_single_session

# mapping of parameter names from numbered to lettered indices (e.g. C_13 to C_OA)
param_mapping = {'C_12':'C_OL', 'C_13':'C_OA', 'C_21':'C_LO', 'C_24':'C_LP', 'C_31':'C_AO', 'C_34':'C_AP', 'C_42':'C_PL', 'C_43':'C_PA',
                 'eta_C_13':'eta_OA', 'eta_C_24':'eta_LP', 'sigma_C_13':'sigma_OA', 'sigma_C_24':'sigma_LP', 'sigma':'sigma', 'G':'G',
                 'patients':'patients', 'controls':'controls'}

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
    fname = 'kdes_rww4D_OU_HPC_20230510_rww4D_OU_HPC_20230605_20230919.pkl'
    with open(os.path.join(proj_dir, 'postprocessing', fname), 'rb') as f:
        kdes = pickle.load(f)
    return kdes


def load_dist_to_FC_controls(args):
    """ Load distances in FC space of data (control & patients) to controls from clinical trial (PRE and POST) """ 
    fname = fname= os.path.join(proj_dir, 'postprocessing', 'distances_to_FC_controls_20230907.pkl')
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
    """ Compute euclidian distance between single empirical FC and simulated FC """ 
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
            fname = os.path.join(proj_dir, 'postprocessing', 'assoc_distances100eps'+str(int(args.tolerance*100))+"_"+ses+today()+".pkl")
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
    with sl.connect(os.path.join(proj_dir, 'postprocessing', 'sim_test_20230614.db')) as conn:
        df = pd.read_sql("SELECT * FROM SIMTEST WHERE test_param='None'", conn)
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
    

def rmse(u,v):
    """ compute the root mean squared error of correlation accross pathways P between u and v as 
    :math:`d = \sqrt{ \sum_{p \in P} (\mu_u^p - \mu_v^p)^2 + (\sigma_u_p - \sigma_v^p)^2}` 
    
    Parameters:
    -----------
    u,v
        pandas DataFrames with only pathway columns

    Returns:
    --------
    d
        Root Mean Squared Error 
    """
    u_ = u.apply([np.mean, np.std])
    v_ = v.apply([np.mean, np.std])
    mse = u_.combine(v_, np.subtract).apply('square').apply('sum').sum()
    return np.sqrt(mse)

metric = {'rmse': rmse, 'emd':emd}

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
    continuous restoration from patients to controls """
    max_len = np.min([len(df_sims), len(df_base[df_base.base_cohort=='controls'])])
    n_folds = int(np.floor(max_len/ args.n_sims))
    n_pathways = len(args.pathways)
    i_s = np.arange(0,n_folds*args.n_sims, args.n_sims)
    outputs = []
    for i,j in itertools.product(i_s, i_s):
        sim = df_sims.iloc[i:i+args.n_sims]
        base = df_base[df_base.base_cohort=='controls'].iloc[j:j+args.n_sims]
        
        # get FC distance from controls FC
        distance = metric[args.distance_metric](sim[args.pathways], base[args.pathways])

        # get parameter difference from patients parameters
        output = dict(('z_'+param, statsmodels.stats.weightstats.ztest(x1=sim[param], 
                                                                       x2=df_base[df_base.base_cohort=='patients'][param])[0]) 
                      for param in args.params)
        
        output['dist'] = distance
        outputs.append(output)
    return outputs


def compute_distance_restore(df_sims, args):
    """ Compare base simulations (from patients and controls) to intervention simulations ("restoration") """
    test_params = df_sims[df_sims.n_test_params<=args.n_test_params].test_param.unique()
    df_base = get_df_base(args)
    df_base = fix_df_base(df_base)
    distance_outputs = Parallel(n_jobs=args.n_jobs, verbose=3)(delayed(compute_distance_restore_sims)
                                        (df_base=df_base, 
                                        df_sims=df_sims[(df_sims.base_cohort=='patients') 
                                                         & (df_sims.test_cohort=='controls') 
                                                         & (df_sims.test_param==test_param)], 
                                         args=args) 
                                        for test_param in test_params)

    #rmses = []
    #for test_param in test_params:
    #    rmses.append(compute_rmse_restore_sims(df_base=df_base[df_base.base_cohort=='controls'], 
    #                              df_sims=df_sims[(df_sims.base_cohort=='patients') 
    #                                               & (df_sims.test_cohort=='controls') 
    #                                                & (df_sims.test_param==test_param)], 
    #                             args=args))

    lines = []
    for tp,outputs in zip(test_params, distance_outputs):
        df_test_param = pd.DataFrame(outputs)
        df_test_param['test_param'] = tp
        df_test_param['mean'] = np.mean(df_test_param.dist)
        df_test_param['median'] = np.median(df_test_param.dist)
        df_test_param['std'] = np.std(df_test_param.dist)
        lines.append(df_test_param)
        #for dist in distance:
        #    lines.append({'test_param':tp, 'dist':dist, 'mean':np.mean(distance), 'median':np.median(distance), 'std':np.std(distance)})
    return pd.concat(lines, ignore_index=True) # <- df_restore

"""
def compute_rmse_restore(df_data, df_sims, args):
    ''' Loop over all the combinations of test_params to compute the RMSE '''
    test_params = df_sims.test_param.unique()
    rmses = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(compute_rmse_restore_data)
                                        (df_data=df_data[df_data.cohort=='controls'], 
                                         df_sims=df_sims[(df_sims.base_cohort=='patients') 
                                                         & (df_sims.test_cohort=='controls') 
                                                         & (df_sims.test_param==test_param)], 
                                         args=args) 
                                         for test_param in test_params)
    lines = []
    for tp,rmse in zip(test_params, rmses):
        for rms in rmse:
            lines.append({'test_param':tp, 'rmse':rms, 'mean':np.mean(rmse), 'std':np.std(rmse)})
    return pd.DataFrame(lines) # <- df_restore
"""    

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
        return np.sqrt(np.sum(np.array(x)-np.array(con_ref))**2)
    def dist_pat(x):
        return np.sqrt(np.sum(np.array(x)-np.array(pat_ref))**2)
    
    distances = {'con':[], 'pat':[], 'con_pat':[]}
    distances['con'] = cons[args.pathways].apply(dist_con, axis=1)
    distances['pat'] = pats[args.pathways].apply(dist_pat, axis=1)
    distances['con_pat'] = pats[args.pathways].apply(dist_con, axis=1)
    return distances


def get_max_distance_sims(args):
    """ Compute the distance between controls and patients in simulated dataset """
    df_base = get_df_base(args)
    df_base = fix_df_base(df_base)
    #n_pathways = len(df.pathway.unique())

    cons = df_base[df_base.base_cohort=='controls']
    pats = df_base[df_base.base_cohort=='patients']

    # using combinations and within cohort stats
    distances = {'con':[], 'pat':[], 'con_pat':[]}
    i_s = list(itertools.islice(range(len(cons)), 0, None, args.n_sims))
    for i,j in itertools.combinations(i_s, 2):
        cons_i = cons.iloc[i:i+args.n_sims]
        cons_j = cons.iloc[j:j+args.n_sims]
        pats_i = pats.iloc[i:i+args.n_sims]
        pats_j = pats.iloc[j:j+args.n_sims]
        distances['con'].append(metric[args.distance_metric](cons_i[args.pathways], cons_j[args.pathways]))
        distances['pat'].append(metric[args.distance_metric](pats_i[args.pathways], pats_j[args.pathways]))
        distances['con_pat'].append(metric[args.distance_metric](cons_i[args.pathways], pats_j[args.pathways]))
    return distances


def plot_efficacy_transform(args):
    """ Plot distance and efficacy distrib and KDEs"""
    
    distances = get_max_distance_sims(args)

    def get_KDE(values):
        vmin, vmax = np.min(values), np.max(values)
        bw = (vmax-vmin)/10
        kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(values).reshape(-1,1))
        X = np.linspace(vmin-3*bw, vmax+3*bw, 100).reshape(-1,1)
        pdf = kde.score_samples(X)
        return {'kde':kde, 'pdf':pdf, 'X':X}
    
    kde_con = get_KDE(distances['con'])
    kde_pat = get_KDE(distances['pat'])
    bias_pat = np.mean(distances['pat'])
    bias_con = np.mean(distances['con'])
    bias = np.mean([bias_con, bias_pat])
    offset = np.mean(distances['con_pat'])
    scale = np.mean(distances['con_pat'])

    plt.figure(figsize=[10,5])
    ax = plt.subplot(2,1,1)
    plt.hist(distances['pat']-np.mean(distances['pat'])+offset, color='orange', alpha=0.5, density=False, bins=np.linspace(0,0.3,30))
    plt.hist(distances['con']-np.mean(distances['con']), bins=np.linspace(0,0.3,30), color='lightblue', alpha=0.5, density=False)
    #plt.xlim([0,1])
    plt.xlabel('$d_Z$', fontsize=12)
    plt.ylabel('counts', fontsize=12)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax = plt.subplot(2,1,2)
    plt.plot(100*(kde_pat['X']-bias)/scale, kde_pat['pdf'], color='orange')
    plt.plot(100*(kde_con['X']-bias+offset)/scale, kde_con['pdf'], color='lightblue')
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


def compute_efficacy(df_restore, base='sims', args=None):
    """ create a new column with treatment efficacy based on distance to controls (in sims or data accroding to argument 'base') 
        Note that simulation metric is the Wassertein distance and the data metric is the euclidiean distance. You have to make
        comparision in FC space using the correct metric. 
    """
    if base=='sims':
        distances = get_max_distance_sims(args)
        offset = np.mean([np.mean(distances['pat']), np.mean(distances['con'])])
        scale = np.mean(distances['con_pat'])
        df_restore['efficacy'] = (1-((df_restore['dist']-offset))/scale)*100 # make % of restoration
    else:
        distances = get_max_distance_data(args)
        offset = np.mean([np.mean(distances['pat']), np.mean(distances['con'])])
        scale = np.mean(distances['con_pat'])
        df_restore['efficacy'] = (1-((df_restore['dist']))/scale)*100 # make % of restoration
    
    #df_top['restore'] = (1-((df_top['dist']-np.mean(distances['pat']))/np.mean(distances['con_pat'])))*100 # make % of restoration
    
    df_restore['n_test_params'] = df_restore.test_param.apply(lambda pars_str: len(pars_str.split(' ')) if pars_str!='None' else 0)
    return df_restore


def get_df_top(sub_df_restore, args):
    # get top parameters that restore FC
    top_params = dict()
    top_params['all'] = sub_df_restore.sort_values('mean').test_param.unique()[:args.n_restore]
    top_params['by_n'] = [sub_df_restore[sub_df_restore.n_test_params==n].sort_values('median').test_param.unique()[:args.n_tops]
                          for n in np.arange(1,args.n_test_params+1)]
    top_params['by_n'] = np.concatenate(top_params['by_n'])
    df_top = sub_df_restore[sub_df_restore.test_param.apply(lambda x: x in top_params[args.sort_style])]   
    return df_top, top_params

def plot_distance_restore(df_restore, args):
    """ plot best FC restoration outputs """
    # adds n_test_params column to dataframe 
    sub_df_restore = df_restore[df_restore.n_test_params<=args.n_test_params]

    # get top parameters that restore FC
    df_top, top_params = get_df_top(df_restore, args)
    
    # normalize distances to get efficacy 
    distances = get_max_distance_sims(args)
    offset = np.mean([np.mean(distances['pat']), np.mean(distances['con'])])
    scale = np.mean(distances['con_pat'])
    #df_top['restore'] = (1-((df_top['dist']-np.mean(distances['pat']))/np.mean(distances['con_pat'])))*100 # make % of restoration
    #df_top['efficacy'] = (1-((df_top['dist']-offset))/scale)*100 # make % of restoration

    # create dfs for base distances (controls and patient with interventions)
    df_pat = pd.DataFrame({'dist': distances['pat']-np.mean(distances['pat']), 'test_param':'patients', 'n_test_params': 0}) # test_param=patients for legend label
    df_pat['efficacy'] = ((df_pat['dist']/np.mean(distances['con_pat'])))*100 # make % of restoration

    df_con = pd.DataFrame({'dist': distances['con']-np.mean(distances['con']), 'test_param':'controls', 'n_test_params': 0}) # test_param=controls for legend label
    df_con['efficacy'] = ((df_con['dist']/np.mean(distances['con_pat']))+1)*100 # make % of restoration
    
    # merge base and restore dataframes
    df_top = pd.concat([df_pat, df_con, df_top], ignore_index=True)
    top_params[args.sort_style] = np.concatenate([['patients','controls'],top_params[args.sort_style]])

    # plotting
    #df_top['restore'] = -((df_top['rmse']/get_max_rmse_data(df_data))-1)*100 # make % of restoration
    palette = {0: 'white', 1: 'lightpink', 2: 'plum', 3: 'mediumpurple', 4: 'lightsteelblue', 5:'skyblue', 6:'royalblue'}
    plt.rcParams.update({'mathtext.default': 'regular', 'font.size':10})
    plt.rcParams.update({'text.usetex': False})
    lw=1
    if args.sort_style == 'all':
        fig = plt.figure(figsize=[5, int(args.n_restore/3)])
    else:
        fig = plt.figure(figsize=[7, int(args.n_tops*args.n_test_params/4)])
        
    sbn.boxplot(data=df_top, x='efficacy', y='test_param', order=top_params[args.sort_style], hue='n_test_params', orient='h', 
                saturation=3, width=0.5, whis=2, palette=palette, dodge=False, linewidth=lw, fliersize=2)
    #sbn.violinplot(df_top, x='restore', y='test_param', order=top_params, hue='n_test_params', orient='h', saturation=3, width=3)
    
    # data and simulated references
    xmin,xmax = plt.gca().get_xlim()
    ymin,ymax = plt.gca().get_ylim()
    
    #plt.grid(axis='x')
    plt.vlines(0, ymin=ymin, ymax=ymax, linestyle='dashed', color='red', alpha=0.75, linewidth=lw) 
    plt.vlines(100, ymin=ymin, ymax=ymax, linestyle='dashed', color='blue', alpha=0.75, linewidth=lw) 

    plt.vlines((np.std(distances['pat'])/np.mean(distances['con_pat']))*100, ymin=ymin, ymax=ymax, 
               linestyle='dashed', color='red', alpha=0.5, linewidth=lw)
    plt.vlines((2*np.std(distances['pat'])/np.mean(distances['con_pat']))*100, ymin=ymin, ymax=ymax, 
               linestyle='dashed', color='red', alpha=0.25, linewidth=lw)

    plt.vlines((1 - (np.std(distances['con'])/np.mean(distances['con_pat'])))*100, ymin=ymin, ymax=ymax, 
               linestyle='dashed', color='blue', alpha=0.5, linewidth=lw)
    plt.vlines((1 - (2*np.std(distances['con'])/np.mean(distances['con_pat'])))*100, ymin=ymin, ymax=ymax, 
               linestyle='dashed', color='blue', alpha=0.25, linewidth=lw)
    
    labels = plt.gca().get_yticklabels()
    new_labels = format_labels(labels)
    plt.gca().set_yticklabels(new_labels)
    plt.gca().spines.top.set_visible(False)
    plt.gca().spines.right.set_visible(False)
    
    sbn.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
    
    if args.save_figs:
        #plt.rcParams['svg.fonttype'] = 'none'
        fname = 'restoration_best_efficacy'+today()+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))

    plt.show(block=False)
    return fig


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
        fname = 'restoration_param_importances'+today()+'.svg'
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


def compute_scaled_feature_score(df_top, params, kdes, scaling='contribution', args=None):
    """ Compute feature scores scaled by their locaion on the KDEs distribution """
    lines = []
    for n_test_params in np.arange(args.n_test_params)+1:
        df_ = df_top[df_top.n_test_params==n_test_params]
        print("Compute param sensitivity for n_test_params={}".format(n_test_params))
        rows = []
        for i,row in df_.iterrows():
            new_row = scale_efficacy_to_kdes(row, params, kdes, scaling)
            rows.append(new_row)
        #rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(scale_efficacy_to_kdes)(row,params,kdes) for _,row in df_.iterrows())
        line = pd.DataFrame(rows).mean(axis=0).to_frame().transpose()
        line['n_test_params'] = n_test_params
        lines.append(line)
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
        fname = 'normalized_params_contribution'+today()+'_small.svg'
        plt.savefig(os.path.join(proj_dir, 'img', fname))
    plt.show()

    
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
        fname = 'normalized_params_log_sensitivity'+today()+'.svg'
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
    """ plot behavioral relationship to distance to FC controls """
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
        behav_diff = df_subj[df_subj.ses=='ses-post'][behav].iloc[0] - df_subj[df_subj.ses=='ses-pre'][behav].iloc[0]
        param_diff = df_subj[df_subj.ses=='ses-post'][param].iloc[0] - df_subj[df_subj.ses=='ses-pre'][param].iloc[0]
        behav_diffs[group].append(behav_diff)
        param_diffs[group].append(param_diff)
        behav_diffs['both'].append(behav_diff)
        param_diffs['both'].append(param_diff)
        if param_diff < 0:
            #plt.scatter(behav_diff, param_diff, color='gray', alpha=0.5)
            responders['behav'][group].append(behav_diff)
            responders['param'][group].append(param_diff)
            responders['behav']['both'].append(behav_diff)
            responders['param']['both'].append(param_diff)
            lines.append({'subj':subj, 'group':group, 'param':param, 'behav':behav, 'param_diff':param_diff, 'behav_diff':behav_diff})
        #else:
            #plt.scatter(param_diff, behav_diff, color=colors[group], alpha=0.2)

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

def score_improvement(df, params, kdes, feature=None):
    """ Score parameters based on improvement in FC space """
    lines = []
    for subj in df.subj.unique():
        df_subj = df[df.subj==subj]

        if feature==None:
            diff_feat = 1
        else:
            diff_feat = df_subj[df_subj.ses=='ses-post'][feature].iloc[0] - df_subj[df_subj.ses=='ses-pre'][feature].iloc[0]
        
        pre = get_param_zscore(df_subj[df_subj.ses=='ses-pre'], params, kdes)
        post = get_param_zscore(df_subj[df_subj.ses=='ses-post'], params, kdes)
        diff_params = post-pre
        
        score = diff_params*diff_feat
        score_dict = dict((par,val) for par,val in zip(params,score))
        score_dict['subj'] = subj
        score_dict['group'] = df_subj.group.unique()[0]
        lines.append(score_dict)
    df_improvement = pd.DataFrame(lines)
    return df_improvement


def plot_improvement_windrose(df_improvement, params, feature=None, gs=None, args=None):
    """ Make windrose vizualisation of improvement in parameter-feature measure space """
    theta = np.arange(len(params))/(len(params)) * 2*np.pi
    theta = np.append(theta, theta[0])

    palette = {'group1': 'orange', 'group2': 'green'}

    if gs==None:
        fig, ax = plt.subplots(1,1, subplot_kw={'projection': 'polar'}, figsize=[3,3])
    else:
        ax = plt.subplot(gs, projection='polar')
    
    #for i,group in enumerate(df_improvement.group.unique()):
    sub_df = df_improvement#[df_improvement.group==group]
    r = np.array(sub_df[params].sum(axis=0))/len(sub_df)
    
    r = np.append(r, r[0]) 
    
    #axes[0,i].plot(theta, r, label = str(df_dt.iloc[i].n_test_params), color=palette[df_dt.iloc[i].n_test_params], lw=5)
    #ax.bar(theta, r, label=group, color=palette[group], lw=5, width=0.5, alpha=0.3)
    for theta_, r_ in zip(theta, r):
        if r_ > 0:
            # increase of parameter due to treatment 
            #ax.bar(theta_, np.abs(r_), color='red', lw=5, width=0.5, alpha=0.2+np.abs(r_)/2)
            ax.bar(theta_, np.abs(r_), color='gray', width=0.5, alpha=0.2+np.abs(r_)/2, lw=1, edgecolor='k', linestyle='-')
        else:
            # decreae of paramter due to treatment
            #ax.bar(theta_, np.abs(r_), color='blue', lw=5, width=0.5, alpha=0.2+np.abs(r_)/2)
            ax.bar(theta_, np.abs(r_), color='gray', width=0.5, alpha=0.2+np.abs(r_)/2, lw=1, edgecolor='k', linestyle='--')

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
            fname = 'improvement_params'+today()+'.svg'
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
    """ Plot pre and post distributions of parameters from digital twin analysis """
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
        print("{:15}: normality={:1}/{:1}   d={: 1.2}    u={:5}  p={:.4f}    t={: .2f}  p={:.4f}   T={: }  p={:.4f}".format(param, p_pre>0.05, p_post>0.05,d, int(u), p, t, pt,int(T), pT))
        #plt.title("u={}  p={:.3f}".format(int(u),p))
        ttl = ''
        if ((p_pre>0.05) and (p_post>0.05)):
            if p<.05:
                ttl = ttl+'*'
            if p*len(params)<0.05:
                ttl = ttl+'*'
        else:
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
            fname = 'distrib_params'+today()+'.svg'
            plt.savefig(os.path.join(proj_dir, 'img', fname))

    plt.tight_layout()
    plt.show()


def plot_improvement_pre_post_params_paper(df_summary, params, gs=None, args=None):
    """ Plot pre and post distributions of parameters from digital twin analysis (trimmed for paper)"""
    i_params = [1,2,5,7] # indices of params to plot

    if gs==None:
        fig = plt.figure(figsize=[6,2])
        sub_gs = plt.GridSpec(nrows=1, ncols=len(i_params)+1)
    else:
        sub_gs = gs.subgridspec(nrows=1, ncols=len(i_params)+1, width_ratios=[1,1,1,0.5,1])

    for i,i_param in enumerate(i_params):
        param = params[i_param]
        #plt.subplot(1, 4, i+1)
        if param.startswith('eta_'):
            ax = plt.subplot(sub_gs[0,i+1])
        else:
            ax = plt.subplot(sub_gs[0,i])
        df_sum = df_summary.melt(id_vars=['subj', 'ses'], value_vars=param, var_name='param', ignore_index=True)
        #sbn.swarmplot(data=df_sum, x='param', y='value', hue='ses', hue_order=['ses-pre', 'ses-post'], dodge=True, 
        #              palette={'ses-pre':'orange', 'ses-post':'purple'}, size=3, alpha=0.5, ax=ax)
        for subj in df_sum.subj.unique():
            sbn.pointplot(data=df_sum[df_sum.subj==subj], y='value', x='ses', order=['ses-pre', 'ses-post'], dodge=True, 
                          color='gray', linewidth=0.5, marker={'size':1}, size=0.5, alpha=0.5, ax=ax)
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
        print("{:15}: d={: 1.2}    u={:5}  p={:.4f}    t={: .2f}  p={:.4f}   T={: }  p={:.4f}".format(param, d, int(u), p, t, pt,int(T), pT))
        #plt.title("u={}  p={:.3f}".format(int(u),p))
        ttl = ''
        if p<.05:
            ttl = ttl+'*'
        if p*len(params)<0.05:
            ttl = ttl+'*'
        ttl = plt.title(ttl, fontsize=14)
        x,y = ttl.get_position()
        ttl.set_position([x,y-0.2])
        #plt.ylabel("${}$".format(format_param(param)), fontsize=12)
        plt.xticks([0,1], labels=['pre', 'post'])
        plt.xlabel("${}$".format(format_param(param)), fontsize=12)
        plt.ylabel('')

        if param.startswith('C_'):
            plt.ylim([-0.12, 0.55])
            if i>0:
                plt.gca().spines.left.set_visible(False)
                plt.yticks([], labels=None)
                plt.subplots_adjust(left=0, right=0.8)
        elif param.startswith('eta_'):
            plt.ylim([-0.01, 0.11])
            plt.yticks([0, 0.05, 0.1])
            plt.subplots_adjust(left=0.2, right=1)
    plt.tight_layout()
    
    if gs==None:
        plt.tight_layout()
        if args.save_figs:
                fname = 'distrib_params_paper'+today()+'.svg'
                plt.savefig(os.path.join(proj_dir, 'img', fname))
        plt.show()


def plot_summary_improvement(df_summary, df_improvement, params, feature=None, args=None):
    """ Just a layout organization to have all plots in same figure """
    fig = plt.figure(figsize=[6,5])
    gs = plt.GridSpec(nrows=2,ncols=4, height_ratios=[1,1])

    # FC - YBOCS scatter + regrression plot
    plot_pre_post_dist_ybocs(df_summary, gs=gs[0,0:2], args=args)

    # Param contribution 
    plot_improvement_windrose(df_improvement, params=params, gs=gs[0,2:4], args=args)

    # pre-post change in param space
    plot_improvement_pre_post_params_paper(df_summary, params=params, gs=gs[1,:], args=args)

    plt.tight_layout()

    if args.save_figs:
        fname = os.path.join(proj_dir, 'img', 'summary_improvement'+today()+'.svg')
        plt.savefig(fname)
    
    plt.show()



def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--save_summary', default=False, action='store_true', help='save output summary')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--n_batch', type=int, default=10, action='store', help="batch size")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    
    parser.add_argument('--db_names', type=str, nargs='+', default=["sim_pat_20230628", "sim_pat_20230721"], help="identifier of the sqlite3 database (use sim_pat_20230628 for digital twin, sim_pat_20230721 for restoration analysis")
    parser.add_argument('--load_sims', default=False, action='store_true', help='Load simulations based on db_names')
    
    parser.add_argument('--base_cohort', type=str, default='controls', help="Cohort from which to infer posterior as default")
    parser.add_argument('--test_cohort', type=str, default='patients', help="Cohort from which to infer posterior of individual params")
    parser.add_argument('--test_params', nargs='+', default=[], help="posterior parameter to swap between base and test cohort, if empty list then all params are tested")
    parser.add_argument('--tolerance', type=float, default=0.05, action='store', help="maximal distance allow to take into consideration 'digital sigling'")
    parser.add_argument('--tolerance_plot', type=float, default=0.3, action='store', help="maximal distance allow to take into consideration 'digital sigling' (for plotting) ")
    parser.add_argument('--save_distances', default=False, action='store_true', help='save distances between patients and simulations')
    parser.add_argument('--load_distances', default=False, action='store_true', help='load distances between patients and simulations from previously saved')
    parser.add_argument('--compute_distances', default=False, action='store_true', help='compute distances between patients and simulations')
    parser.add_argument('--n_closest', type=int, default=1, action='store', help="number of digital twins to retain sorted by increasing distance")
    parser.add_argument('--plot_param_behav', default=False, action='store_true', help='plot param-behavioral relationship')
    parser.add_argument('--verbose', default=False, action='store_true', help='print extra processing info')
    parser.add_argument('--session', default=None, action='store', help='which session (ses-pre or ses-post) for behavioral scores (default:None => both are used')
    parser.add_argument('--plot_efficacy_transform', default=False, action='store_true', help='plot the transformation from distance from controls to efficacy')
    
    parser.add_argument('--multivariate_analysis', default=False, action='store_true', help='perform multivariate analysis on simulations parameters')
    parser.add_argument('--multivar_fc', default=False, action='store_true', help='perform multivariate analysis on FC variables')
    parser.add_argument('--cv_type', type=str, default='LeaveOneOut', help="which cross-validation scheme to apply (KFold, RepeatedKFold, ShuffleSplit, LeaveOneOut)")
    parser.add_argument('--n_splits', type=int, default=10, help="number of splits used for cross-validation")
    parser.add_argument('--n_repeats', type=int, default=5, help="number of repetitions for the RepeatedKFold cross-validation")
    parser.add_argument('--test_size', type=float, default=0.3, help="ratio of test data (over all data) used for ShuffleSplit cross-validation")
    parser.add_argument('--plot_cv_regression', default=False, action='store_true', help='plot cross validation regression scatters')
    parser.add_argument('--plot_multivar_svd', default=False, action='store_true', help='plot dimensionality reduction on regression coefficients')
    parser.add_argument('--plot_multivariate_results', default=False, action='store_true', help='plot multivariate linear regression coefficients')
    
    parser.add_argument('--null', default=False, action='store_true', help='shuffle coupling weights to create null hypothesis for regression coefficients')
    parser.add_argument('--n_null', type=int, default=100, action='store', help="number of elements to make null distribution")
    parser.add_argument('--plot_null_distrib', default=False, action='store_true', help='plot null distribution analysis of linear regression coefficients')
    parser.add_argument('--plot_cv_null_distrib', default=False, action='store_true', help='plot null distribution analysis of linear regression coefficients for all CV folds')
    
    parser.add_argument('--print_ANOVA', default=False, action='store_true', help='print stats for mixed and multiple one-way ANOVAs')
    
    parser.add_argument('--restore_analysis', default=False, action='store_true', help='perform retoration analys of test parameters to move from patient to controls FC')
    parser.add_argument('--n_restore', type=int, default=10, action='store', help="number of best restorations for plotting")
    parser.add_argument('--n_tops', type=int, default=5, action='store', help="number of best restorations for each n_test_param for plotting")
    parser.add_argument('--n_test_params', type=int, default=6, action='store', help="max number of parameter combinations for plotting restoration outputs")
    parser.add_argument('--distance_metric', type=str, default='rmse', help="distance used in restoration metric (rmse or emd)")
    parser.add_argument('--sort_style', type=str, default='by_n', help="how to sort distances for visualization: 'by_n' or 'all' (default)")
    parser.add_argument('--max_depth', type=int, default=3, action='store', help="max depth of the decision tree")
    parser.add_argument('--plot_distance_restore', default=False, action='store_true', help='plot efficacy horizontal box plots')
    
    parser.add_argument('--contribution_sensitivity_analysis', default=False, action='store_true', help='run ontribution and sensitivity analysis')
    parser.add_argument('--load_param_contribution', default=False, action='store_true', help='load parameter contributions from previously computed restoration analysis')
    parser.add_argument('--load_param_sensitivity', default=False, action='store_true', help='load parameter sensitivity from previously computed restoration analysis')
    parser.add_argument('--plot_params_contribution_sensitivity', default=False, action='store_true', help='plot contribution and sensitivity of parameters in restoration analysis')

    parser.add_argument('--predictive_analysis', default=False, action='store_true', help='Analyse predictive power of model based on distance to controls FC')
    parser.add_argument('--plot_fc_dist_pre_post_behav', default=False, action='store_true', help='plot FC distance to controls in pre-post TMS data')
    parser.add_argument('--compute_post_distances', default=False, action='store_true', help='Compute distances for digital twins analysis using post-TMS FC')
    parser.add_argument('--compute_sim_vecs', default=False, action='store_true', help='Compute simulation vectors in FC space (otherwise load pre-computed)')
    parser.add_argument('--save_sim_vecs', default=False, action='store_true', help='Save simulation vectors in FC space after being computed')
    parser.add_argument('--load_sim_vecs', default=False, action='store_true', help='Load simulation vectors in FC space')
    parser.add_argument('--load_post_distances', default=False, action='store_true', help='load distances in post-TMS inference')
    parser.add_argument('--plot_pre_post_associations', default=False, action='store_true', help='plot pre-post TMS associations between FC, params and behavioral measures')

    parser.add_argument('--load_summary', default=False, action='store_true', help='load summary of digital twin analysis')
    parser.add_argument('--improvement_analysis', default=False, action='store_true', help='Analyse improvement of models parameters pre-post TMS')
    parser.add_argument('--param_improvement', type=str, default=None, help="defines which parameter to scale improvement with")
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
        fname = os.path.join(proj_dir, 'postprocessing', args.db_namess[0]+'_distances100eps'+str(int(args.tolerance*100))+".pkl")
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

        df_restore = compute_distance_restore(df_sims[df_sims.test_param!='None'], args)
        df_restore = compute_efficacy(df_restore, args=args)
        df_top, top_params = get_df_top(df_restore, args)
        if args.plot_distance_restore:
            plot_distance_restore(df_restore, args=args)

        #df_feature_importance, decision_trees, feat_imps = decision_tree(df_restore, params, args)
        #df_custom_feat_imps = compute_custom_feature_scores(decision_trees, params, args)
        #df_simple_feat_imps = compute_simple_feature_scores(df_top, params, args)

        #df_reliability = compute_feature_reliability(df_top, params, kdes, args)
        #plot_contribution_windrose(df_reliability, params=['z_'+param for param in params], args=args)

        df_scaled_efficacy = compute_scaled_feature_score(df_restore[df_restore.efficacy>0], params, kdes, scaling='contribution', args=args)
        plot_contribution_windrose(df_scaled_efficacy, params, args=args)

        # Parameters contribution
        if args.contribution_sensitivity_analysis:
            if args.load_param_contribution:
                with open(os.path.join(proj_dir, 'postprocessing', 'df_param_contribution.pkl'), 'rb') as f:
                    df_params_contribution = pickle.load(f)
            else:
                df_params_contribution = compute_scaled_feature_score(df_restore[df_restore.efficacy>0], params, kdes, scaling='contribution', args=args)
                if args.save_outputs:
                    with open(os.path.join(proj_dir, 'postprocessing', 'df_param_contribution.pkl'), 'wb') as f:
                        pickle.dump(df_params_contribution, f)        
            if args.plot_params_contribution_sensitivity:
                plot_contribution_windrose(df_params_contribution, params, args=args)

            # Parameters sensitivity
            if args.load_param_sensitivity:
                with open(os.path.join(proj_dir, 'postprocessing', 'df_param_sensitivity.pkl'), 'rb') as f:
                    df_param_sensitivity = pickle.load(f)
            else:
                df_param_sensitivity = compute_scaled_feature_score(df_restore[df_restore.efficacy>0], params, kdes, scaling='sensitivity', args=args)
                if args.save_outputs:
                    with open(os.path.join(proj_dir, 'postprocessing', 'df_param_sensitivity.pkl'), 'wb') as f:
                        pickle.dump(df_param_sensitivity, f)        
            if args.plot_params_contribution_sensitivity:
                plot_sensitivity_windrose(df_param_sensitivity, params, args=args)
        

        


    # prediction 
    if args.predictive_analysis:
        print('Load functional distances (data)...')
        df_fc_pre_post = load_dist_to_FC_controls(args)

        # delta FC vs delta YBOCS
        df_dist_fc = df_fc_pre_post.merge(df_fc_pat, on=['subj', 'ses', 'group', *pathways], how='inner')    
        df_dist_fc = drop_single_session(df_dist_fc)
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
            fname = os.path.join(proj_dir, 'postprocessing', args.db_names[0]+'_distances100eps'+str(int(args.tolerance*100))+"_post.pkl")
            with open(fname, 'rb') as f:
                assoc_post = pickle.load(f) 
        
        df_sim_post = merge_data_sim_dfs(df_pat[df_pat.ses=='ses-post'], df_sims, assoc_post, args)
        df_post = pd.merge(df_fc_pre_post[df_fc_pre_post.ses=='ses-post'], df_sim_post, on=['subj', 'ses', 'group'], how='outer', suffixes=[None, '_sim'])


        # pre-post analysis summary
        df_pre = pd.merge(df_fc_pre_post[df_fc_pre_post.ses=='ses-pre'], 
                          df_sim_pre, 
                          on=['subj', 'ses', 'group'], how='outer', suffixes=[None, '_sim'])
        if args.save_summary:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_pre'+today()+".pkl"), 'wb') as f:
                pickle.dump(df_pre, f)
            with open(os.path.join(proj_dir, 'postprocessing', 'df_post'+today()+".pkl"), 'wb') as f:
                pickle.dump(df_post, f)
        
        df_summary = df_post[df_post.cohort=='patients'].merge(df_pre[df_pre.cohort=='patients'], how='outer')
        if args.save_summary:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_summary'+today()+".pkl"), 'wb') as f:
                pickle.dump(df_summary, f)
            
    
        if args.plot_pre_post_associations:
            plot_fc_dist_pre_post_behav(df_summary)
            plot_fc_dist_pre_post_params(df_summary)
            plot_pre_post_params_behavs(df_summary)
    

    if args.load_summary:
        # df_summary_20230915.pkl : pre and post with controls/patients param swaps
        # df_summary_20230918.pkl : only post with param swaps
        with open(os.path.join(proj_dir, 'postprocessing', 'df_summary_20230918.pkl'), 'rb') as f:
                df_summary = pickle.load(f)

    if args.improvement_analysis:
        #plot_pre_post_dist_ybocs(df_summary, args=args)
        df_improvement = score_improvement(df_summary, params, kdes)
        plot_summary_improvement(df_summary, df_improvement, params, args=args)
        #plot_improvement_windrose(df_improvement, params, args=args)
        
        #for param_imp in [None]:
        #    df_improvement = score_improvement(df_summary, params, kdes, feature=param_imp)
        #    plot_improvement_windrose(df_improvement, params, feature=param_imp, args=args)
            #plot_improvement_distrib(df_improvement, param_imp=param_imp)
            #print_stats_improvement(df_improvement, params, param_imp=param_imp)
            #plot_improvement_pre_post_params(df_summary, params, args)
        
        #plot_improvement_pre_post_params_paper(df_summary, params, args)


