### Analyze history of optimizations
##  Author: Sebastien Naze
#   QIMR 2023

import argparse
from datetime import datetime 
import itertools
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os 
import pandas as pd
import pickle
import pingouin as pg
import pyabc
import scipy
import seaborn as sbn
import sklearn

import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils import cohen_d, get_working_dir, today, proj_dir

#working_dir = get_working_dir()
#proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')


def import_results(args, config=None):
    """ Read optimization results from DB """ 
    if (args.histories==None and config!=None):
        print("Reading history from configuration file, if you want to add history in command line, \
              use --histories db_filename1 db_filename2, etc.")
        
        args.histories = [config['optim_params']['db_name']+'optim']
        args.gens = [config['optim_params']['max_nr_populations']]
        args.history_names = [config['optim_params']['cohort']]

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


def get_ax_inds(nrows, ncols, row_offset=2, col_offset=2):
    """ create list of axes to iterate from when plotting """
    for row_ind, col_ind in itertools.product(np.arange(nrows), np.arange(ncols)):
        if ((col_ind<col_offset) & (row_ind<row_offset)):
            continue
        else:
            yield (row_ind, col_ind)


def plot_epsilons(histories, ax=None, args=None):
    """ Plot evolution of epsilons across generations 
    
    Parameters
    ----------
        histories: list
            Optimization pyABC history objects (e.g. controls and patients)
        ax: matplotlib.Axes
            (optional) Axis to draw the figure.
        args: argparse.Namespace
            (optional) Extra arguments (for saving, etc).

    """ 
    n_hist = len(histories)
    if ax==None:
        fig = plt.figure(figsize=[5, 5])
        ax = plt.subplot(1,1,1)

    for i,(name,history) in enumerate(histories.items()):
        if name=='controls':
            pyabc.visualization.plot_epsilons([history], ax=ax)
        plt.legend(list(histories.keys()))


def plot_weights(histories, gs=None, nrows=None, ncols=None, row_offset=None, col_offset=None, args=None):
    """ Plot evolution of weights across generations
    
    Parameters
    ----------
        histories: list
            Optimization pyABC history objects (e.g. controls and patients)
        gs: matplotlib.GridSpec
            (optional) GridSpec object to draw the figure into.
        nrows, ncols: int
            number of rows/columns to make the grid.
        row_offset, col_offset: int
            offsets to account for when some (top left) entries of the grid should be left empty.
        args: argparse.Namespace
            (optional) Extra arguments (e.g. for saving, etc).

    """ 
    n_hist = len(histories)
    n_gen = np.max([h.max_t for h in histories.values()])
    
    if gs==None:
        fig = plt.figure(figsize=[15, 4])
        nrows,ncols, row_offset, col_offset = 0,0,0,0
        gs = plt.GridSpec(nrows=nrows, ncols=ncols)
        
    
    for i,(name,history) in enumerate(histories.items()):
        ax_inds = get_ax_inds(nrows, ncols, row_offset=row_offset, col_offset=col_offset)
        #for j in np.arange(history.max_t+1):
        for j in np.arange(1,10):
            x,y = next(ax_inds)
            ax = plt.subplot(gs[x,y])
            df,w  = history.get_distribution(t=j)
            
            plt.scatter(np.arange(len(w)), w, s=10, alpha=0.1)
            plt.title(f"t={j}", fontsize=10)
            plt.yscale('log')
            #if x==nrows-1:
            plt.xlabel('particle', fontsize=10)
            if (((y==col_offset) and (x<row_offset)) or (y==0)):
                plt.ylabel('$\omega$', fontsize=10)
            #plt.legend(args.history_names)
            ax.spines.top.set_visible(False)
            ax.spines.right.set_visible(False)
            #xtl = ax.get_xticklabels()
            #ax.set_xticklabels(xtl, fontsize=12)
            #ytl = ax.get_yticklabels()
            #ax.set_yticklabels(ytl, fontsize=12)
    plt.tight_layout()


def plot_param_distrib(histories, args):
    """ plot posterior distribution at end of optimization """
    colors = {'controls':'lightskyblue', 'patients':'orange'}
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
                plt.hist(df[col], alpha=0.2, color=colors[args.history_names[i]])
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

    Parameters
    ---------
        histories: list
            Controls and patient pyABC history objects. 
        args: argparse.Namespace
            Extra options

    Returns
    -------
        df_stats: pandas DataFrame 
            Statistics.  

    """
    df_post_con,w_con = histories['controls'].get_distribution(t=args.gens[0])
    df_post_pat,w_pat = histories['patients'].get_distribution(t=args.gens[1])
    cols = df_post_con.columns
    mc = len(cols)
    stats = []
    for col in cols:
        x,y = df_post_con[col], df_post_pat[col]
        
        stat_norm_con,p_norm_con = scipy.stats.normaltest(x)
        stat_norm_pat,p_norm_pat = scipy.stats.normaltest(y)

        t,p_t = scipy.stats.ttest_ind(x,y, permutations=10000)
    
        u,p_u = scipy.stats.mannwhitneyu(x,y)
        
        h,p_h = scipy.stats.kruskal(x,y)

        d = OCD_modeling.utils.cohen_d(x,y)

        ks_res = scipy.stats.ks_2samp(x,y)
        
        line = "{:15}    OCD={:.2f}     controls={:.2f}     t={:8.2f}    p={:.4f}    p_bf={:.4f}    normality(con/pat)={}/{}    \n\
                U={:8d}    p={:.4f}    p_bf={:.4f}    H={:8.3f}    p={:.4f}    p_bf={:.4f}    \
                d={:.4f}    ks={:.2f}    p_ks={:.4f}    p_bf={:.4f}".format(
            col,y.median(),x.median(),t,p_t,p_t*mc,p_norm_con>0.05,p_norm_pat>0.05,int(u),p_u, p_u*mc,h,p_h,p_h*mc,d,ks_res.statistic, ks_res.pvalue, ks_res.pvalue*mc)
        if args.plot_stats:
            print(line)
        
        df_line = {'param':col, 't':t, 'p_t':p_t, 'p_t_bf':p_t*len(cols), 'normality':(p_norm_con>0.05)&(p_norm_pat>0.05), \
                'u':u, 'p_u':p_u, 'p_u_bf':p_u*len(cols), \
                'h':h, 'p_h':p_h, 'p_h_bf':p_h*len(cols), 'd':d, 'ks':ks_res.statistic, 'p_ks':ks_res.pvalue}
        stats.append(df_line)
    df_stats = pd.DataFrame(stats)
    return df_stats


def compute_kdes(histories, n_pts = 100, args=None):
    """ 
    Computes Kernel Density Estimates (KDEs) of the posterior distributions of parameters.
    
    Parameters
    ----------
        histories: dict
            Nested dictionnary of SQL alchemy history objects.
        n_pts: int 
            Number of points used to estimate the probability density functions (PDFs).
        args: argparse.Namespace
            Extra options.

    Returns
    -------
        kdes: dict 
            Nested dictiorany of KDEs and associated PDFs.
        cols: list 
            List of parameters for which the KDEs were estimated.
    
    """

    kdes = dict()

    cols = []
    for i,(cohort,history) in enumerate(histories.items()):
        kdes[cohort] = dict()
        df,w = history.get_distribution(t=args.gens[i])
        for col in df.columns:
            cols.append(col)
            vmin, vmax = df[col].min(), df[col].max()
            bw = (vmax-vmin)/10
            kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(df[col]).reshape(-1,1))
            X = np.linspace(vmin-0.5*bw, vmax+0.5*bw, 100).reshape(-1,1)
            pdf = kde.score_samples(X)
            kdes[cohort][col] = {'kde':kde, 'pdf':pdf, 'X':X, 'vals': df[col]}
    cols = np.unique(cols)

    if args.save_kdes:
        fname = os.path.join(proj_dir, 'postprocessing', 'kdes_'+'_'.join(args.histories)+today()+'.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(kdes, f)

    return kdes, cols

def custom_plot_epsilons(histories, ax=None, n_gens=11):
    colors = ['lightblue', 'orange']
    labels = ['controls', 'OCD']#list(histories.keys())

    if ax==None:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)

    for i,(name,history) in enumerate(histories.items()):
        # extract epsilons
        # note: first entry is from calibration and thus translates to inf,
        # thus must be discarded
        eps = np.array(history.get_all_populations()['epsilon'][:n_gens])

        # plot
        ax.plot(eps, 'x--', label=labels[i], color=colors[i], alpha=1)
        
    plt.xticks(np.arange(0,n_gens,2), labels=np.arange(0,n_gens,2))
    plt.yscale('log')
    plt.ylabel('$\epsilon$', fontsize=10)
    plt.xlabel('$t$', fontsize=10)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    plt.legend(labels, fontsize=10)
    #xtl = ax.get_xticklabels()
    #ax.set_xticklabels(xtl, fontsize=12)
    #ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

# -------------------------------------
# Observed vs simulated stats and plots
# -------------------------------------

def stats_obs_vs_sim(df_data, df_base):
    """ T-tests within and across cohorts, observed and simulated traces  """ 
    # prep sim dataframe
    df_sim = df_base.melt(id_vars=['subj', 'base_cohort'], value_vars=['Acc_OFC', 'Acc_PFC', 'Acc_dPut', 'OFC_PFC', 'dPut_OFC', 'dPut_PFC'], var_name='pathway', value_name='corr')
    df_sim = df_sim.rename(columns={'base_cohort':'cohort'})

    df_data['type'] = 'observed'
    df_sim['type'] = 'simulated'
    df_fc = pd.concat([df_data, df_sim], join='inner', ignore_index=True)

    pathways = ['Acc_OFC', 'Acc_PFC', 'Acc_dPut', 'OFC_PFC', 'dPut_OFC', 'dPut_PFC']
    stats = dict((pathway,dict()) for pathway in pathways)
    print("================")
    print("TWO-FACTOR ANOVA")
    print("================")
    for pathway in df_fc.pathway.unique():
        print(pathway)
        aov = pg.anova(data=df_fc[df_fc.pathway==pathway], between=['cohort', 'type'], dv='corr')
        pg.print_table(aov)
        stats[pathway]['aov'] = aov

    print("=============")
    print("WELCH T-TESTS")
    print("=============")
    for pathway in pathways:
        print(pathway)
        df_tmp = df_fc[df_fc.pathway==pathway]
        cc = pg.ttest(df_tmp[(df_tmp.cohort=='controls') & (df_tmp.type=='observed')]['corr'], df_tmp[(df_tmp.cohort=='controls') & (df_tmp.type=='simulated')]['corr'])
        print("con-con -- obs-sim   T={:.2f}   p={:.4f}   d={:.2f}".format(cc['T'][0], cc['p-val'][0], cc['cohen-d'][0]))
        pp = pg.ttest(df_tmp[(df_tmp.cohort=='patients') & (df_tmp.type=='observed')]['corr'], df_tmp[(df_tmp.cohort=='patients') & (df_tmp.type=='simulated')]['corr'])
        print("pat-pat -- obs-sim   T={:.2f}   p={:.4f}   d={:.2f}".format(pp['T'][0], pp['p-val'][0], pp['cohen-d'][0]))
        cpo = pg.ttest(df_tmp[(df_tmp.cohort=='controls') & (df_tmp.type=='observed')]['corr'], df_tmp[(df_tmp.cohort=='patients') & (df_tmp.type=='observed')]['corr'])
        print("con-pat -- obs-obs   T={:.2f}   p={:.4f}   d={:.2f}".format(cpo['T'][0], cpo['p-val'][0], cpo['cohen-d'][0]))
        cps = pg.ttest(df_tmp[(df_tmp.cohort=='controls') & (df_tmp.type=='simulated')]['corr'], df_tmp[(df_tmp.cohort=='patients') & (df_tmp.type=='simulated')]['corr'])
        print("con-pat -- sim-sim   T={:.2f}   p={:.4f}   d={:.2f}".format(cps['T'][0], cps['p-val'][0], cps['cohen-d'][0]))
        os = pg.ttest(df_tmp[(df_tmp.type=='observed')]['corr'], df_tmp[(df_tmp.type=='simulated')]['corr'])
        print("obs-sim   T={:.2f}   p={:.4f}   d={:.2f}".format(cps['T'][0], cps['p-val'][0], cps['cohen-d'][0]))
        print("\n")

        stats[pathway]['cc'], stats[pathway]['pp'], stats[pathway]['cpo'], stats[pathway]['cps'], stats[pathway]['os'] = cc, pp, cpo, cps, os

    return stats

def plot_fc_sim_vs_data(df_data, df_base, stats, axes=None, args=None):
    """ Plot functional connectivity (Pearson correlation) across the frontostriatal regions of interests 
    in empirical and simulated data. 
    
    Parameters
    ----------
        df_data: pandas.DataFrame
            Empirical data extracted from fMRI in OCD subjects and healthy controls.
        df_base: pandas.DataFrame
            Simulated data using parameters infered from posterior distributions (either only OCD parameters  
            or only controls, no permutation to model virtual intervention).
        stats: dict
            (deprecated) Statistics within and between cohorts for both empirical and simulated data (deprecated 
            since now stats are directly computed within plotting function).
        axes: list of matplotlib.Axes
            (optional) List of axes to plot data (if embedded in other figure, otherwise create new figure).
        args: argparse.Namespace
            (optional) Extra arguments (e.g. for saving, etc).
    
    """ 
    
    palette = {'controls': 'lightblue', 'patients': 'orange'}
    pathways = ['Acc_OFC', 'Acc_PFC', 'Acc_dPut', 'OFC_PFC', 'dPut_OFC', 'dPut_PFC']
    pathway_names = ['NAcc-OFC', 'NAcc-LPFC', 'NAcc-dPut', 'OFC-LPFC', 'dPut-OFC', 'dPut-LPFC']
    df_tmp = df_base.iloc[:384].melt(id_vars=['subj', 'base_cohort'], value_vars=pathways, var_name='pathway', value_name='corr')
    df_tmp = df_tmp.rename(columns={'base_cohort':'cohort'})

    # select n random simulations for each cohort
    n = 50 
    cons = df_tmp[df_tmp.cohort=='controls'].subj.unique()
    cons = cons[np.random.permutation(len(cons))][:n]
    pats = df_tmp[df_tmp.cohort=='patients'].subj.unique()
    pats = pats[np.random.permutation(len(pats))][:n]
    subjs = np.concatenate([cons, pats])
    df_tmp = df_tmp[df_tmp.subj.isin(subjs)]
    
    if axes==None:
        plt.figure(figsize=[10,3])
        ax_data = plt.subplot(1,2,1)
        ax_sim = plt.subplot(1,2,2)
        plt.rcParams.update({'figure.autolayout':True})
        plt.tight_layout()
    else:
        ax_data = axes['data']
        ax_sim = axes['sim']


    def plot_stats_stars(df, ax, y_level=0.5):
        """ Add stars on top of plots for stats significance """ 
        for i,pathway in enumerate(pathways):
            ttest = scipy.stats.ttest_ind(np.array(df[(df.pathway==pathway) & (df.cohort=='controls')]['corr']), 
                                np.array(df[(df.pathway==pathway) & (df.cohort=='patients')]['corr']))
            if ttest.pvalue<(0.05/len(pathways)):
                ax.plot(i,y_level, '*', color='black')

    # DATA
    # ----
    sbn.swarmplot(data=df_data, x='pathway', y='corr', hue='cohort', order=pathways, 
                            ax=ax_data, marker='o', size=2.8, dodge=True, palette=palette, 
                            hue_order=['controls', 'patients'])
    #plt.xticks(rotation=30, ha='right')
    for label in ax_data.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('center')
        label.set_va('top')
        label.set_rotation_mode('default') #'anchor'
    ax_data.legend().set_visible(False)
    ax_data.spines.top.set_visible(False)
    ax_data.spines.right.set_visible(False)
    plot_stats_stars(df_data, ax=ax_data, y_level=0.45)
    ax_data.set_title('Observed')
    ax_data.set_ylim([-0.35, 0.5])
    ax_data.set_ylabel('R', fontsize=12)
    #ax_data.set_xlabel('Pathway', fontsize=12)
    ax_data.set_xticklabels(pathway_names)


    # SIMULATION
    # ----------
    #sbn.swarmplot(data=df_tmp, x='pathway', y='corr', hue='base_cohort', 
    #                       ax=ax_sim, size=1.5, dodge=True, palette=palette, hue_order=['controls', 'patients'])
    sbn.swarmplot(data=df_tmp, x='pathway', y='corr', hue='cohort', 
                           ax=ax_sim, size=2.8, dodge=True, palette=palette, hue_order=['controls', 'patients'])
    #plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
    for label in ax_sim.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('center')
        label.set_va('top')
        label.set_rotation_mode('default') #'anchor'
    ax_sim.legend().set_visible(False)
    ax_sim.spines.top.set_visible(False)
    ax_sim.spines.right.set_visible(False)
    ax_sim.spines.left.set_visible(False)
    ax_sim.set_yticks([])
    plot_stats_stars(df_tmp, ax=ax_sim, y_level=0.45)
    ax_sim.set_title('Simulated')
    ax_sim.set_ylim([-0.35, 0.5])
    ax_sim.set_ylabel('')
    ax_sim.set_yticklabels([])
    #ax_sim.set_xlabel('Pathway', fontsize=12)
    ax_sim.set_xticklabels(pathway_names)

    if args.save_figs:
        today = datetime.now().strftime('_%Y%m%d')
        fname = os.path.join(proj_dir, 'img', 'fc_sim_vs_data'+today+'.svg')
        plt.savefig(fname)
    
    

def plot_kdes(kdes, cols, df_stats, df_real=[], df_pred=[], 
              plot_args={'nrows':4, 'ncols':5, 'row_offset':2, 'col_offset':3, 'figsize':[10,7], 'show_stars':True, 'hist_alpha':0.3, 'kde_alpha':1}, 
              args=None):
    """  Plot Kernel Density Estimates of posteriors (controls vs OCD)
    
    Parameters
    ----------
        kdes: dict
            Kernel Density Estimates of parameters
        cols: list
            Model parameters
        df_stats: pandas.DataFrame
            Stastics for each parameter (healthy controls vs OCD patients)
        df_real: pandas.DataFrame
            (Optional) Synthetic data (observed)
        df_pred: pandas.DataFrame
            (Optional) Synthetic data (predicted)
        plot_args: dict
            Default options for plotting.
                nrows, ncols: number of rows and columns of the GridSpec object (grid of axes).
                row_offset, col_offset: shifts in rows and colums to let space for another plot.
                figsize: figure size.
                show_stars: show stars for statistical significant between controls and OCD.
                hist_alpha, kde_alpha: opacity of the histograms and kernel density estimates.
        args: argparse.Namespace
            (Optional) Extra arguments

    """

    plt.rcParams.update({'mathtext.default': 'regular', 'font.size':10})
    plt.rcParams.update({'text.usetex': False})
    plt.rcParams.update({'figure.constrained_layout.use': False})

    nrows, ncols = plot_args['nrows'], plot_args['ncols']
    row_offset, col_offset = plot_args['row_offset'], plot_args['col_offset']

    fig = plt.figure(figsize=plot_args['figsize'])
    gs = plt.GridSpec(nrows=nrows, ncols=ncols, hspace=0.7, wspace=0.25)
    
    # EPSILONS
    #ax = fig.add_subplot(gs[0:row_offset, 0:col_offset])
    #custom_plot_epsilons(histories, ax=ax)

    # subject vertical lines colors:
    subjs_colors = ['yellowgreen', 'plum']
    
    # KDES
    ax_inds = get_ax_inds(nrows, ncols, row_offset, col_offset)
    for _,col in enumerate(cols):
        #ax = plt.subplot(3,5,i+2)
        i,j = next(ax_inds)
        ax = fig.add_subplot(gs[i,j])
        vals = np.concatenate([kdes['controls'][col]['vals'], kdes['patients'][col]['vals']])
        bins = np.linspace(vals.min(), vals.max(), 13)
        plt.hist(kdes['controls'][col]['vals'], color='lightblue', bins=bins, alpha=plot_args['hist_alpha'], density=True, log=False, lw=0.3, ec='gray')
        plt.hist(kdes['patients'][col]['vals'], color='orange', bins=bins, alpha=plot_args['hist_alpha'], density=True, log=False, lw=0.3, ec='gray')
        plt.plot(kdes['controls'][col]['X'], np.exp(kdes['controls'][col]['pdf']), 'lightblue', lw=1, alpha=plot_args['kde_alpha'])
        plt.plot(kdes['patients'][col]['X'], np.exp(kdes['patients'][col]['pdf']), 'orange', lw=1, alpha=plot_args['kde_alpha'])
        xlabl = OCD_modeling.mcmc.inference_analysis.format_labels([matplotlib.text.Text(text=col)], args.N)[0]
        plt.xlabel(xlabl, fontsize=12)
        if j==0:
            plt.ylabel("$\\rho$", fontsize=12)
        else:
            plt.ylabel('')
        #ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        mn,mx = ax.get_ylim()
        if mn<0.01:
            ax.set_ylim(bottom=0.01)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        
        ttl=''
        if plot_args['show_stars']:
            if df_stats[df_stats.param==col]['p_u_bf'].iloc[0]<0.05:
            #if df_stats[df_stats.param==col]['p_t_bf'].iloc[0]<0.05:
                ttl+='*'
                if np.abs(df_stats[df_stats.param==col]['d'].iloc[0])>0.2:
                    ttl+='*'
                    if np.abs(df_stats[df_stats.param==col]['d'].iloc[0])>0.5:
                        ttl+='*'
        plt.title(ttl, fontsize=14)

        if len(df_real)!=0:
            for i,subj_df in enumerate(df_real):
                plt.vlines(subj_df[col], ymin=mn, ymax=mx, color=subjs_colors[i], lw=2)
                if args.annotate_restore:
                    ctl_mean = kdes['controls'][col]['vals'].mean()
                    arrow_height = mx - (i+1)*5*(mx-mn)/100  # 5% below top and by step of 5% for each subjs
                    plt.vlines(ctl_mean, ymin=mn, ymax=mx, color='black', linewidth=3, alpha=0.5)
                    arrow = matplotlib.patches.FancyArrowPatch((float(subj_df[col]),arrow_height), (ctl_mean,arrow_height), 
                                                    arrowstyle='->,head_width=0.1,head_length=0.1', mutation_scale=20, edgecolor=subjs_colors[i], linewidth=2)
                    ax.add_patch(arrow)
        
        if len(df_pred)!=0:
            plt.vlines(df_pred[col], ymin=mn, ymax=mx, color='red')


    # FC
    #axes = {'data': fig.add_subplot(gs[3:, 0:3]),
    #        'sim': fig.add_subplot(gs[3:, 3:])}
    #plot_fc_sim_vs_data()

    if args.annotate_restore:
        ax = fig.add_subplot(gs[0,0])
        for i,subj_df in enumerate(df_real):
            ax.text(0.5,-i*0.5, subj_df.iloc[0]['subj'], backgroundcolor=matplotlib.colors.colorConverter.to_rgba(subjs_colors[i], 0.3), horizontalalignment='center')
            ax.text(0.5,-i*0.5-0.25, 'vs', horizontalalignment='center')
        ax.text(0.5,-len(df_real)*0.5, 'Healthy Average', fontweight='bold', horizontalalignment='center')
        ax.set_ylim([-1,0.2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout(pad=0)
    
    if args.save_figs:
        today = datetime.now().strftime('_%Y%m%d')
        fname = os.path.join(proj_dir, 'img', 'kdes'+today+'.svg')
        plt.savefig(fname)
    plt.show()
    return fig
    

# ------------------------------------------ #
# Simulate at posterior distributions' modes #
# ------------------------------------------ #

def get_kde_peaks(kdes):
    """ Extract peaks of posterior distribution from kernel density estimates """
    peaks = dict()
    for cohort in kdes.keys():
        peaks[cohort]= dict()
        for param in kdes[cohort].keys():
            i_max = np.argmax(kdes[cohort][param]['pdf'])
            X_max = kdes[cohort][param]['X'][i_max]
            peaks[cohort][param] = X_max
    return peaks


def simulate_posterior_modes(peaks, args):
    """ Run simulations for posterior distribution's modes  """
    sim_outputs = dict()
    for cohort, params in peaks.items():
        sim_args = argparse.Namespace()
        sim_args.n_sims = args.n_sims
        sim_args.n_jobs= args.n_jobs
        sim_args.t_tot = 15000
        sim_args.t_range = [5000, 15000]
        sim_args.model_pars, sim_args.sim_pars, sim_args.control_pars, sim_args.bold_pars = OCD_modeling.mcmc.unpack_params(params)

        sim_objs = OCD_modeling.hpc.launch_simulations(sim_args)
        sim_outputs[cohort] = sim_objs
    return sim_outputs


def plot_epsilons_weights(histories, args):
    """ Combine the epsilons tolerances and the optimization weights """
    fig = plt.figure(figsize=[12,4])
    nrows, ncols = 2,5
    row_offset, col_offset = 1,1
    gs = plt.GridSpec(nrows=nrows, ncols=ncols)
    ax = fig.add_subplot(gs[:row_offset,:col_offset])
    custom_plot_epsilons(histories, ax=ax, n_gens=11)
    plot_weights(histories, gs=gs, nrows=nrows, ncols=ncols, row_offset=row_offset, col_offset=col_offset, args=args)
    
    if args.save_figs:
        fname = os.path.join(proj_dir, 'img', 'epsilons_weights_log'+today()+'.svg')
        plt.savefig(fname)
        fname = os.path.join(proj_dir, 'img', 'epsilons_weights_log'+today()+'.png')
        plt.savefig(fname)
        fname = os.path.join(proj_dir, 'img', 'epsilons_weights_log'+today()+'.pdf')
        plt.savefig(fname)
    plt.show()


def get_history_parser():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType('rb'), help='Project config file to use (TOML file).')

    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--gens', nargs='+', default=[], action='store', help="generation of the optimization (list, must be same length as histories)")
    parser.add_argument('--histories', nargs='+', default=None, action='store', help="optimizations to analyse and compare")
    parser.add_argument('--history_names', type=list, default=['controls', 'patients'], action='store', help="names given to each otpimization loaded")
    parser.add_argument('--compute_stats', default=False, action='store_true', help='compute statistics of parameter differnces between groups')
    parser.add_argument('--compute_kdes', default=False, action='store_true', help='compute KDEs of parameter estimations, needs compute_stats to run.')

    parser.add_argument('--save_kdes', default=False, action='store_true', help='save KDEs')
    parser.add_argument('--save_kdes_suffix', type=str, default=today(), help="identifier of the KDE in the saving folder")
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')

    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--plot_epsilons', default=False, action='store_true', help='plot optimization errors')
    parser.add_argument('--plot_weights', default=False, action='store_true', help='plot optimization weights for each gen')
    parser.add_argument('--plot_epsilons_weights', default=False, action='store_true', help='plot optimization errors')
    parser.add_argument('--plot_param_distrib', default=False, action='store_true', help='plot parameter distribution for each gen')
    parser.add_argument('--plot_kde_matrix', default=False, action='store_true', help='plot KDE matrix of optimization')
    parser.add_argument('--plot_stats', default=False, action='store_true', help='show statisics')
    parser.add_argument('--plot_kdes', default=False, action='store_true', help='plot KDEs of optimzed params')
    parser.add_argument('--plot_fc_sim_vs_data', default=False, action='store_true', help='plot FC of inference from optimization vs real data')
    parser.add_argument('--simulate_posterior_modes', default=False, action='store_true', help='run simulations with at posterior distributions modes ')

    parser.add_argument('--N', type=int, default=4, action='store', help="number of node in the soumilations (default N=4, i.e. no thalamus, other option is N=6 (with thalamus)")
    
    parser.add_argument('--annotate_restore', default=False, action='store_true', help='Annotate the plots with arrow between subjects params and the avg controls')
    return parser


if __name__=='__main__':
    args = get_history_parser().parse_args()
    config = OCD_modeling.utils.read_config(args.config)
    
    histories = import_results(args, config)
    
    if args.plot_epsilons:
        #plot_epsilons(histories, args)
        custom_plot_epsilons(histories, n_gens=np.array(args.gens, dtype=int).max())
    if args.plot_weights:
        plot_weights(histories, args)
    if args.plot_epsilons_weights:
        plot_epsilons_weights(histories, args)
    if args.plot_param_distrib:
        plot_param_distrib(histories, args)
    if args.plot_kde_matrix:
        plot_kde_matrix(histories, args)


    # posterior distributions stats
    if args.compute_stats:
        df_stats = compute_stats(histories, args)

    # Kernel Density Estimates
    if args.compute_kdes:
        kdes,cols = compute_kdes(histories, args=args)

    if args.plot_kdes:
        plot_args = {4:{'nrows':4, 'ncols':5, 'row_offset':2, 'col_offset':3, 'figsize':[10,7], 'show_stars':True, 'hist_alpha':0.3, 'kde_alpha':1},
                     6:{'nrows':6, 'ncols':5, 'row_offset':0, 'col_offset':0, 'figsize':[10,12], 'show_stars':True, 'hist_alpha':0.3, 'kde_alpha':1}}
        plot_kdes(kdes, cols, df_stats, plot_args=plot_args[args.N], args=args)

    if args.plot_fc_sim_vs_data:
        # load simulated and observed data
        df_base = OCD_modeling.mcmc.get_df_base(args)
        #df_base = OCD_modeling.mcmc.fix_df_base(df_base)
        df_data = OCD_modeling.mcmc.load_df_data(args)
    
        # observed vs simulated stats
        stats = stats_obs_vs_sim(df_data, df_base)

        plot_fc_sim_vs_data(df_data, df_base, stats, args=args)

    if args.simulate_posterior_modes:
        kde_peaks = get_kde_peaks(kdes)
        sim_outputs = simulate_posterior_modes(kde_peaks, args)
        if args.save_outputs:
            with open(os.path.join(proj_dir, 'postprocessing', 'sims_posterior_modes'+today()+'.pkl'), 'wb') as f:
                pickle.dump(sim_outputs, f)
