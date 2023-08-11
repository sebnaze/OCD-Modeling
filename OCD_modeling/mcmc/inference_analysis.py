### Analyse inferred simulations w.r.t. observed data
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
from joblib import Parallel, delayed
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


#-------------------------#
#   Digital twin analysis #
#-------------------------#

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
    patients = np.sort(df_data[df_data.cohort=='patients'].subj.unique())

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
        fname = os.path.join(proj_dir, 'postprocessing', args.db_name+'_distances100eps'+str(int(args.tolerance*100))+".pkl")
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
    df_sims['subj'] = df_sims.apply(lambda row: "sim-"+args.base_cohort[:3]+"{:06d}".format(int(row.name)+1), axis=1)


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


def plot_cv_regression(multivar, args=None):
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


def compute_rmse_restore_sims(df_base, df_sims, args):
    """ Compute Root Mean Squared Error between batches of simulations in controls, patients and 
    continuous restoration from patients to controls """
    max_len = np.min([len(df_sims), len(df_base)])
    n_folds = int(np.floor(max_len/ args.n_sims))
    n_pathways = len(args.pathways)
    i_s = np.arange(0,n_folds*args.n_sims, args.n_sims)
    RMSEs = []
    for i,j in itertools.product(i_s, i_s):
        sim = df_sims.iloc[i:i+args.n_sims]
        base = df_base.iloc[j:j+args.n_sims]
        RMSEs.append(rmse(sim[args.pathways], base[args.pathways]))
    return np.array(RMSEs).squeeze()


def compute_rmse_restore(df_sims, args):
    """ Compare base simulations (from patients and controls) to intervention simulations ("restoration") """
    test_params = df_sims.test_param.unique()
    df_base = get_df_base(args)
    df_base = fix_df_base(df_base)
    rmses = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(compute_rmse_restore_sims)
                                        (df_base=df_base[df_base.base_cohort=='controls'], 
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
    for tp,rmse in zip(test_params, rmses):
        for rms in rmse:
            lines.append({'test_param':tp, 'rmse':rms, 'mean':np.mean(rmse), 'std':np.std(rmse)})
    return pd.DataFrame(lines) # <- df_restore

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

def get_max_rmse_data(df_data):
    """ Compute the RMSE between controls and patients in real data """
    RMSE = []
    for pathway in df_data.pathway.unique():
        con = df_data[(df_data.cohort=='controls') & (df_data.pathway==pathway)]['corr']
        pat = df_data[(df_data.cohort=='patients') & (df_data.pathway==pathway)]['corr']
        RMSE.append((con.mean()-pat.mean())**2 + (con.std()-pat.std())**2)
    return np.sqrt(np.sum(RMSE))


def get_max_rmse_sims(args):
    """ Compute the RMSE between controls and patients in simulated dataset """
    df = get_df_base(args)
    n_pathways = len(df.pathway.unique())

    cons = df[df.base_cohort=='controls']
    pats = df[df.base_cohort=='patients']

    # using combinations and within cohort stats
    rmse_combi = {'con':[], 'pat':[], 'con_pat':[]}
    i_s = list(itertools.islice(range(len(cons)), 0, None, args.n_sims*6))
    for i,j in itertools.combinations(i_s, 2):
        con_mean_i = cons.iloc[i:i+args.n_sims*n_pathways].groupby('pathway')['corr'].mean()
        con_std_i = cons.iloc[i:i+args.n_sims*n_pathways].groupby('pathway')['corr'].std()
        pat_mean_i = pats.iloc[i:i+args.n_sims*n_pathways].groupby('pathway')['corr'].mean()
        pat_std_i = pats.iloc[i:i+args.n_sims*n_pathways].groupby('pathway')['corr'].std()

        con_mean_j = cons.iloc[j:j+args.n_sims*n_pathways].groupby('pathway')['corr'].mean()
        con_std_j = cons.iloc[j:j+args.n_sims*n_pathways].groupby('pathway')['corr'].std()
        pat_mean_j = pats.iloc[j:j+args.n_sims*n_pathways].groupby('pathway')['corr'].mean()
        pat_std_j = pats.iloc[j:j+args.n_sims*n_pathways].groupby('pathway')['corr'].std()

        rmse_con = np.sqrt(((con_mean_i-con_mean_j)**2).sum() + ((con_std_i-con_std_j)**2).sum())
        rmse_pat = np.sqrt(((pat_mean_i-pat_mean_j)**2).sum() + ((pat_std_i-pat_std_j)**2).sum())
        rmse_con_pat = np.sqrt(((con_mean_i-pat_mean_j)**2).sum() + ((con_std_i-pat_std_j)**2).sum())
        rmse_combi['con'].append(rmse_con)
        rmse_combi['pat'].append(rmse_pat)
        rmse_combi['con_pat'].append(rmse_con_pat)

    return rmse_combi
    

def plot_restore_intersect(rmse):
    """ Plot KDEs of restoration RMSE between controls and patients to find a correct threshold value """
    
    def get_KDE(values):
        vmin, vmax = np.min(values), np.max(values)
        bw = (vmax-vmin)/10
        kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(values).reshape(-1,1))
        X = np.linspace(vmin-3*bw, vmax+3*bw, 100).reshape(-1,1)
        pdf = kde.score_samples(X)
        return {'kde':kde, 'pdf':pdf, 'X':X}
    
    kde_con = get_KDE(rmse['con'])
    kde_pat = get_KDE(rmse['pat'])
    bias_pat = np.mean(rmse['pat'])
    bias_con = np.mean(rmse['con'])
    offset = np.mean(rmse['con_pat'])
    scale = np.mean(rmse['con_pat'])

    plt.figure(figsize=[12,4])
    plt.subplot(2,1,1)
    plt.hist(rmse['pat'], color='orange', alpha=0.5, density=False)
    plt.hist(rmse['con']+offset, color='lightblue', alpha=0.5, density=False)
    plt.subplot(2,1,2)
    plt.plot(100*(kde_pat['X']-bias_pat)/scale, kde_pat['pdf'], color='orange')
    plt.plot(100*(kde_con['X']+offset-bias_con)/scale, kde_con['pdf'], color='lightblue')
    plt.show()


def plot_rmse_restore(df_restore, df_data, args):
    """ plot best FC restoration outputs """
    # adds n_test_params column to dataframe 
    df_restore['n_test_params'] = df_restore.test_param.apply(lambda pars_str: len(pars_str.split(' ')))
    sub_df_restore = df_restore[df_restore.n_test_params<=args.n_test_params]
    # get top parameters that restore FC
    top_params = sub_df_restore.sort_values('mean').test_param.unique()[:args.n_restore]
    df_top = sub_df_restore[sub_df_restore.test_param.apply(lambda x: x in top_params)]

    rmse = get_max_rmse_sims(args)

    # plotting
    #df_top['restore'] = -((df_top['rmse']/get_max_rmse_data(df_data))-1)*100 # make % of restoration
    df_top['restore'] = (1-((df_top['rmse']-np.mean(rmse['pat']))/np.mean(rmse['con_pat'])))*100 # make % of restoration
    plt.figure(figsize=[6, int(args.n_restore/2)])
    sbn.boxplot(df_top, x='restore', y='test_param', order=top_params, hue='n_test_params', orient='h', saturation=3, width=1)
    #plt.xlim([-100,100])
    #sbn.stripplot(df_top, x='test_param', y='mean', color='red', order=top_params)
    #plt.xticks(rotation=90)

    # data and simulated references
    xmin,xmax = plt.gca().get_xlim()
    ymin,ymax = plt.gca().get_ylim()
    #plt.hlines(y=, xmin=xmin, xmax=xmax, linestyle='dashed', color='black', label='data')
    #plt.hlines(y=get_max_rmse_sims(), xmin=xmin, xmax=xmax, linestyle='dashed', color='blue', label='sim')
    #plt.grid(axis='x')
    plt.vlines(0, ymin=ymin, ymax=ymax, linestyle='dashed', color='red', alpha=0.75)
    #plt.vlines(-(((mu_sims-sigma_sims)/mu_sims)-1)*100, ymin=ymin, ymax=ymax, linestyle='dashed', color='blue', alpha=0.5)
    #plt.vlines(-(((mu_sims-2*sigma_sims)/mu_sims)-1)*100, ymin=ymin, ymax=ymax, linestyle='dashed', color='green', alpha=0.5)
    #plt.vlines(-(((get_max_rmse_data(df_data))/mu_sims)-1)*100, ymin=ymin, ymax=ymax, linestyle='dashed', color='red', alpha=0.5) 

    plt.vlines(100, ymin=ymin, ymax=ymax, linestyle='dashed', color='blue', alpha=0.75) 

    plt.vlines((np.std(rmse['pat'])/np.mean(rmse['con_pat']))*100, ymin=ymin, ymax=ymax, linestyle='dashed', color='red', alpha=0.5)
    plt.vlines((2*np.std(rmse['pat'])/np.mean(rmse['con_pat']))*100, ymin=ymin, ymax=ymax, linestyle='dashed', color='red', alpha=0.25)

    plt.vlines((1 - (np.std(rmse['con'])/np.mean(rmse['con_pat'])))*100, ymin=ymin, ymax=ymax, linestyle='dashed', color='blue', alpha=0.5)
    plt.vlines((1 - (2*np.std(rmse['con'])/np.mean(rmse['con_pat'])))*100, ymin=ymin, ymax=ymax, linestyle='dashed', color='blue', alpha=0.25)
    plt.show(block=False)
    

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
    y = df_restore['rmse']

    return np.array(list(X)), np.array(y)


def decision_tree(df_restore, params):
    """ create a tree from restoration output 
    Parameters:
    -----------
    df_restore
        pandas DataFrame of restoration output
    params
        list of model parameters
    """
    X,y = get_X_y(df_restore, params)
    dt = sklearn.tree.DecisionTreeRegressor(max_depth=args.max_depth)
    dt.fit(X, y)

    df_fi = pd.DataFrame([dict((feat, imp) for feat,imp in zip(params, dt.feature_importances_))])
    

    plt.figure(figsize=[6,4])
    sbn.barplot(df_fi)
    plt.xticks(rotation=30)
    plt.show()

    plt.figure(figsize=[40,10])
    sklearn.tree.plot_tree(dt, max_depth=3, feature_names=params, proportion=True)
    plt.show()

def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
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
    parser.add_argument('--compute_distances', default=False, action='store_true', help='compute distances between patients and simulations')
    parser.add_argument('--n_closest', type=int, default=1, action='store', help="batch size")
    parser.add_argument('--plot_param_behav', default=False, action='store_true', help='plot param-behavioral relationship')
    parser.add_argument('--verbose', default=False, action='store_true', help='print extra processing info')
    parser.add_argument('--session', default=None, action='store', help='which session (ses-pre or ses-post) for behavioral scores (default:None => both are used')
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
    parser.add_argument('--n_test_params', type=int, default=4, action='store', help="max number of parameter combinations for plotting restoration outputs")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    # load histories and KDEs
    #histories = import_results(args)
    
    behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total', 'Dep_Total', 'Anx_total']
    params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42', 'eta_C_13', 'eta_C_24', 'sigma', 'sigma_C_13', 'sigma_C_24']

    print("Loading simulations...")
    df_sims = load_df_sims(args)
    
    print("Fixing simulations indices...")
    fix_df_sims_names(df_sims, args)
    
    print("Loading functional connectivity data...")
    df_data = load_df_data(args)
    
    print("Loading behavioral data...")
    with open(os.path.join(proj_dir, 'postprocessing', 'df_pat_.pkl'), 'rb') as f:
        df_pat = pickle.load(f)
        if args.session!=None:
            df_pat = df_pat[df_pat.ses==args.session]
    df_fc_pat = df_data.pivot(columns='pathway', values='corr', index='subj').reset_index().merge(df_pat)

    
    if args.load_distances:
        print("Loading distances...")
        fname = os.path.join(proj_dir, 'postprocessing', args.db_name+'_distances100eps'+str(int(args.tolerance*100))+".pkl")
        with open(fname, 'rb') as f:
            assoc = pickle.load(f)
        df_sim_pat = merge_data_sim_dfs(df_pat, df_sims, assoc, args)
    elif args.compute_distances:
        print("Computing distances...")
        assoc = compute_distances(df_data, df_sims, args)
        df_sim_pat = merge_data_sim_dfs(df_pat, df_sims, assoc, args)
    

    # univariate analysis
    if args.plot_param_behav:
        plot_param_behav(df_sim_pat, params=params, args=args)

    # multivariate analysis
    if args.multivariate_analysis:
        print("Running multivariate analysis...")
        if args.multivar_fc:
            params=['Acc_OFC', 'Acc_PFC', 'Acc_dPut', 'OFC_PFC', 'dPut_PFC']
            multivar = multivariate_analysis(df_fc_pat, 
                                             params=params, 
                                             args=args)
        else:
            multivar = multivariate_analysis(df_sim_pat, params=params, args=args)

        if args.plot_multivariate_results:
            plot_multivariate_results(multivar, args=args)
        
        if args.plot_cv_regression:
            plot_cv_regression(multivar, args=args)
        
        if args.plot_multivar_svd:
            plot_multivar_svd(multivar, behavs=behavs, args=args)

        if args.null:
            print("Creating Null distribution...")
            models = dict(('null_model{:04d}'.format(i), sklearn.linear_model.Ridge(alpha=0.01)) 
                                for i in range(args.n_null))
            multivar_null = multivariate_analysis(df_sim_pat, 
                                                  params=params, 
                                                  models=models, 
                                                  null=True, 
                                                  args=args)

            create_df_null(multivar, multivar_null)

            if args.plot_null_distrib:
                plot_null_distrib(multivar, args)

        if args.print_ANOVA:
            print_ANOVA(df_sim_pat, behavs, params)


    # restoration analysis
    if args.restore_analysis:
        print("Restoration analysis...")
        args.pathways = df_data.pathway.unique()
        #df_restore = compute_rmse_restore(df_data, df_sims, args)
        df_restore = compute_rmse_restore(df_sims, args)
        plot_rmse_restore(df_restore, df_data, args=args)
        decision_tree(df_restore, params)


    
