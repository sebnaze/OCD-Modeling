### Analyse inferred simulations w.r.t. observed data
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor
import copy
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


def score_pval(y, y_pred):
    res = scipy.stats.pearsonr(y, y_pred)
    return res.pvalue


def multivariate_analysis(df_sim_pat, params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42'], 
                          behavs=['YBOCS_Total', 'OCIR_Total', 'OBQ_Total', 'MADRS_Total', 'HAMA_Total', 'Anx_total', 'Dep_Total'], 
                          models={'Ridge':sklearn.linear_model.Ridge(alpha=0.01)}, null=False, args=None):
    """ Multivariate regression of parameters to predict behaviors """
    output = dict()
    p_score = sklearn.metrics.make_scorer(score_pval, greater_is_better=False)
    for i,behav in enumerate(behavs):
        output[behav] = dict()
        # Linear regression
        X = df_sim_pat[params] 
        y = df_sim_pat[behav]
            
        for model_name,model in models.items():
            # create null couplings 
            if null:
                #data=np.array(df_sim_pat[params])
                #data = np.random.permutation(data.ravel()).reshape(data.shape)
                #X = pd.DataFrame(data=data, columns=params)
                data=np.array([np.random.permutation(df_sim_pat[param]) for param in params]).T
                X = pd.DataFrame(data=data, columns=params)
            model.fit(X,y)
            y_pred = model.predict(X)
            mae = sklearn.metrics.median_absolute_error(y, y_pred)
            print("Median Abs Error (MAE) = {:.5f}".format(mae))

            r2 = sklearn.metrics.r2_score(y, y_pred)
            # stats on prediction 
            r,p = scipy.stats.pearsonr(y,y_pred)

            # coefficients' variability 
            #cv = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
            #cv = sklearn.model_selection.KFold(n_splits=5)
            #cv = sklearn.model_selection.ShuffleSplit(n_splits=10,test_size=0.3)
            cv = sklearn.model_selection.LeaveOneOut()
            cv_results = sklearn.model_selection.cross_validate(
                    copy.deepcopy(model),
                    X,
                    y,
                    cv=cv,
                    #scoring={'p':p_score, 'r2':sklearn.metrics.make_scorer(sklearn.metrics.r2_score)},
                    scoring='neg_mean_absolute_error',
                    return_estimator=True,
                    return_train_score=True,
                    return_indices=True,
                    n_jobs=8,
            )

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

        plt.figure(figsize=[15,5])
        for i,train_idx in enumerate(train_ids):
            test_idx = test_ids[i]
            est = multivar[behav]['Ridge']['cv_results']['estimator'][i]

            X_train = df_sim_pat.iloc[train_idx][params]
            X_test = df_sim_pat.iloc[test_idx][params]
            y_train = df_sim_pat.iloc[train_idx][behav]
            y_test = df_sim_pat.iloc[test_idx][behav]
            train_pred = est.predict(X_train)
            test_pred = est.predict(X_test)

            plt.subplot(2,5,i+1)
            plt.scatter(y_train, train_pred, label='train')
            plt.scatter(y_test, test_pred, label='test')
            plt.xlabel(behav+' ground truth')
            plt.ylabel(behav+' prediction')

            r2 = multivar[behav]['Ridge']['cv_results']['test_r2'][i]
            p = multivar[behav]['Ridge']['cv_results']['test_p'][i]
            plt.title("r2={:.2f}  p={:.2f}".format(r2,-p))
        plt.tight_layout()
        plt.show()


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
        for model_name, _ in models.items():
            model = multivar_null[behav][model_name]['model']
            nulls.append(dict((f,v) for f,v in zip(model.feature_names_in_, model.coef_)))
        df_null = pd.DataFrame(nulls)
        df_null['null'] = True

        df_sim = pd.DataFrame(
            [dict((f,v) for f,v in zip(multivar[behav]['Ridge']['model'].feature_names_in_, multivar[behav]['Ridge']['model'].coef_))]
        )
        df_sim['null'] = False

        df = pd.concat([df_null, df_sim], ignore_index=True)

        multivar[behav]['Ridge']['df_null'] = df


def plot_null_distrib(multivar, args=None):
    """ plot null distributions of regression coefficients and stats """
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.size':11, 'axes.titlesize':'medium', 'mathtext.default': 'regular'})

    behavs = list(multivar.keys())
    params = multivar[behavs[0]]['Ridge']['model'].feature_names_in_
    
    # first figure: individual axes for each params + stats
    for j,behav in enumerate(behavs):
        fig = plt.figure(figsize=[15,3])
        df = multivar[behav]['Ridge']['df_null']
        for i,par in enumerate(params):
            ax = fig.add_subplot(1, len(params), i+1)
            n, mu, sigma = len(df[df.null][par]), df[df.null][par].mean(), df[df.null][par].std()
            t = (float(df[~df.null][par]) - mu) / sigma
            p = scipy.stats.t.sf(np.abs(t), n-1)                                        
            sbn.swarmplot(df[df.null][par], ax=ax, color='black', alpha=0.1)
            sbn.swarmplot(df[~df.null][par], ax=ax, color='red', alpha=1)
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

    # second figure: single axe for all params, no stats 
    fig = plt.figure(figsize=[15,3*len(behavs)])
    for j,behav in enumerate(behavs):
        ax = fig.add_subplot(len(behavs), 1, j+1)
        sbn.swarmplot(df.melt(id_vars='null'), x='variable', y='value', hue='null', palette=['black', 'gainsboro'], ax=ax)
        ax.get_legend().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(behav)
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
    parser.add_argument('--session', default=None, action='store', help='which session (ses-pre or ses-post) for behavioral scores (default:None => both are used')
    parser.add_argument('--multivariate_analysis', default=False, action='store_true', help='perform multivariate analysis on simulations parameters')
    parser.add_argument('--multivar_fc', default=False, action='store_true', help='perform multivariate analysis on FC variables')
    parser.add_argument('--plot_cv_regression', default=False, action='store_true', help='plot cross validation regression scatters')
    parser.add_argument('--plot_multivar_svd', default=False, action='store_true', help='plot dimensionality reduction on regression coefficients')
    parser.add_argument('--plot_multivariate_results', default=False, action='store_true', help='plot multivariate linear regression coefficients')
    parser.add_argument('--null', default=False, action='store_true', help='shuffle coupling weights to create null hypothesis for regression coefficients')
    parser.add_argument('--n_null', type=int, default=100, action='store', help="number of elements to make null distribution")
    parser.add_argument('--plot_null_distrib', default=False, action='store_true', help='plot null distribution analysis of linear regression coefficients')
    parser.add_argument('--print_ANOVA', default=False, action='store_true', help='print stats for mixed and multiple one-way ANOVAs')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_arguments()
    # load histories and KDEs
    #histories = import_results(args)
    
    #behavs=['OCIR_Total', 'MADRS_Total', 'HAMA_Total']
    behavs=['OBQ_Total', 'Dep_Total', 'Anx_total']

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

    print("Loading distances...")
    if args.load_distances:
        fname = os.path.join(proj_dir, 'postprocessing', args.db_name+'_distances100eps'+str(int(args.tolerance*100))+".pkl")
        with open(fname, 'rb') as f:
            assoc = pickle.load(f)
    else:
        assoc = compute_distances(df_data, df_sims, args)
    
    df_sim_pat = merge_data_sim_dfs(df_pat, df_sims, assoc, args)
    
    # univariate analysis
    if args.plot_param_behav:
        plot_param_behav(df_sim_pat, args=args)

    # multivariate analysis
    if args.multivariate_analysis:
        print("Running multivariate analysis...")
        if args.multivar_fc:
            params=['Acc_OFC', 'Acc_PFC', 'Acc_dPut', 'OFC_PFC', 'dPut_PFC']
            multivar = multivariate_analysis(df_fc_pat, 
                                             params=params, 
                                             args=args)
        else:
            params=['C_12', 'C_13', 'C_24', 'C_31', 'C_34', 'C_42']
            multivar = multivariate_analysis(df_sim_pat, args=args)

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
            multivar_null = multivariate_analysis(df_sim_pat, models=models, null=True, args=args)

            create_df_null(multivar, multivar_null)

            if args.plot_null_distrib:
                plot_null_distrib(multivar, args)

        if args.print_ANOVA:
            print_ANOVA(df_sim_pat, behavs, params)

    
