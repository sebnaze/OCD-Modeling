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
import pyabc
import scipy
import seaborn as sbn
import sklearn

import OCD_modeling
# import most relevant environment and project variable
from OCD_modeling.utils import cohen_d, get_working_dir, today

working_dir = get_working_dir()
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')


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


def plot_epsilons(histories, fig=None, args=None):
    """ Plot evolution of epsilons across generations """ 
    n_hist = len(histories)
    if fig==None:
        fig = plt.figure(figsize=[5, 5])
        ax = plt.subplot(1,1,1)
    
    # include epsilons to KDE plot
    else:
        ax = fig.add_subplot(3,5,1)

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

        d = OCD_modeling.utils.cohen_d(x,y)

        ks_res = scipy.stats.ks_2samp(x,y)
        
        line = "{:15}    t={:8.2f}    p={:.4f}    p_bf={:.4f}    normality(con/pat)={}/{}    \
                U={:8d}    p={:.4f}    p_bf={:.4f}    H={:8.3f}    p={:.4f}    p_bf={:.4f}    \
                d={:.4f}    ks={:.2f}    p_ks={:.4f}".format(
            col,t,p_t,p_t*mc,p_norm_con>0.05,p_norm_pat>0.05,int(u),p_u, p_u*mc,h,p_h,p_h*mc,d,ks_res.statistic, ks_res.pvalue)
        print(line)
        
        df_line = {'param':col, 't':t, 'p_t':p_t, 'p_t_bf':p_t/len(cols), 'normality':(p_norm_con>0.05)&(p_norm_pat>0.05), \
                'u':u, 'p_u':p_u, 'p_u_bf':p_u/len(cols), \
                'h':h, 'p_h':p_h, 'p_h_bf':p_h/len(cols), 'd':d, 'ks':ks_res.statistic, 'p_ks':ks_res.pvalue}
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
            bw = (vmax-vmin)/10
            kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(df[col]).reshape(-1,1))
            X = np.linspace(vmin-0.5*bw, vmax+0.5*bw, 100).reshape(-1,1)
            pdf = kde.score_samples(X)
            kdes[cohort][col] = {'kde':kde, 'pdf':pdf, 'X':X, 'vals': df[col]}
    cols = np.unique(cols)

    if args.save_kdes:
        fname = os.path.join()

    return kdes, cols

def custom_plot_epsilons(histories, ax, n_gens=11):
    colors = ['lightblue', 'orange']
    labels = list(histories.keys())
    for i,(name,history) in enumerate(histories.items()):
        # extract epsilons
        # note: first entry is from calibration and thus translates to inf,
        # thus must be discarded
        eps = np.array(history.get_all_populations()['epsilon'][1:n_gens])

        # plot
        ax.plot(eps, 'x--', label=labels[i], color=colors[i], alpha=1)
        
    plt.xticks(np.arange(1,10,2), labels=np.arange(1,10,2)+1)
    plt.ylabel('$\epsilon$', fontsize=12)
    plt.xlabel('$t$', fontsize=12)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    plt.legend(list(histories.keys()))




def get_kde_ax_inds(nrows, ncols, row_offset=2, col_offset=2):
    """ create list of axes to iterate from when plotting """
    for row_ind, col_ind in itertools.product(np.arange(nrows), np.arange(ncols)):
        if ((col_ind<col_offset) & (row_ind<row_offset)):
            continue
        else:
            yield (row_ind, col_ind)


def plot_fc_sim_vs_data(axes=None, args=None):
    df_base = OCD_modeling.mcmc.get_df_base(args)
    df_base = OCD_modeling.mcmc.fix_df_base(df_base)
    df_data = OCD_modeling.mcmc.load_df_data(args)
    
    palette = {'controls': 'lightblue', 'patients': 'orange'}
    pathways = ['Acc_OFC', 'Acc_PFC', 'Acc_dPut', 'OFC_PFC', 'dPut_OFC', 'dPut_PFC']
    df_tmp = df_base.iloc[:384].melt(id_vars=['subj', 'base_cohort'], value_vars=pathways, var_name='pathway', value_name='corr')

    if axes==None:
        plt.figure(figsize=[10,3])
        ax_data = plt.subplot(1,2,1)
        ax_sim = plt.subplot(1,2,2)
        plt.rcParams.update({'figure.autolayout':True})
        plt.tight_layout()
    else:
        ax_data = axes['data']
        ax_sim = axes['sim']

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
    ax_data.set_title('Observed')
    ax_data.set_ylim([-0.35, 0.5])
    ax_data.set_ylabel('R', fontsize=12)
    ax_data.set_xlabel('Pathway', fontsize=12)


    # SIMULATION
    # ----------
    sbn.swarmplot(data=df_tmp, x='pathway', y='corr', hue='base_cohort', 
                           ax=ax_sim, size=1.5, dodge=True, palette=palette, hue_order=['controls', 'patients'])
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
    ax_sim.set_title('Simulated')
    ax_sim.set_ylim([-0.35, 0.5])
    ax_sim.set_ylabel('')
    ax_sim.set_yticklabels([])
    ax_sim.set_xlabel('Pathway', fontsize=12)

    if args.save_figs:
        today = datetime.now().strftime('_%Y%m%d')
        fname = os.path.join(proj_dir, 'img', 'fc_sim_vs_data'+today+'.svg')
        plt.savefig(fname)
    
    

def plot_kdes(kdes, cols, histories, args=None):
    """  Plot Kernel Density Estimates of posteriors """

    plt.rcParams.update({'mathtext.default': 'regular', 'font.size':10})
    plt.rcParams.update({'text.usetex': False})
    plt.rcParams.update({'figure.constrained_layout.use': False})

    nrows, ncols = 3,6
    row_offset, col_offset = 2,2

    fig = plt.figure(figsize=[10,5])
    gs = plt.GridSpec(nrows=nrows, ncols=ncols)
    
    # EPSILONS
    ax = fig.add_subplot(gs[0:row_offset, 0:col_offset])
    custom_plot_epsilons(histories, ax=ax)
    
    # KDES
    ax_inds = get_kde_ax_inds(nrows, ncols, row_offset, col_offset)
    for _,col in enumerate(cols):
        #ax = plt.subplot(3,5,i+2)
        i,j = next(ax_inds)
        ax = fig.add_subplot(gs[i,j])
        plt.hist(kdes['controls'][col]['vals'], color='lightblue', bins=10, alpha=0.2, density=True, log=False)
        plt.hist(kdes['patients'][col]['vals'], color='orange', bins=10, alpha=0.2, density=True, log=False)
        plt.plot(kdes['controls'][col]['X'], np.exp(kdes['controls'][col]['pdf']), 'lightblue', lw=1)
        plt.plot(kdes['patients'][col]['X'], np.exp(kdes['patients'][col]['pdf']), 'orange', lw=1)
        xlabl = OCD_modeling.mcmc.inference_analysis.format_labels([matplotlib.text.Text(text=col)])[0]
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
        


    # FC
    #axes = {'data': fig.add_subplot(gs[3:, 0:3]),
    #        'sim': fig.add_subplot(gs[3:, 3:])}
    #plot_fc_sim_vs_data()

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

def parse_arguments():
    " Script arguments when ran as main " 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--n_sims', type=int, default=50, action='store', help="number of simulations ran with the same parameters (e.g. to get distribution that can be campared to clinical observations)")
    parser.add_argument('--gens', type=list, default=[], action='store', help="generation of the optimization (list, must be same length as histories)")
    parser.add_argument('--histories', nargs='+', default=None, action='store', help="optimizations to analyse and compare")
    parser.add_argument('--history_names', type=list, default=['controls', 'patients'], action='store', help="names given to each otpimization loaded")
    parser.add_argument('--compute_kdes', default=False, action='store_true', help='compute KDEs of parameter estimations')
    
    parser.add_argument('--save_kdes', default=False, action='store_true', help='save KDEs')
    parser.add_argument('--save_kdes_suffix', type=str, default=today(), help="identifier of the KDE in the saving folder")
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')

    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--plot_epsilons', default=False, action='store_true', help='plot optimization errors')
    parser.add_argument('--plot_weights', default=False, action='store_true', help='plot optimization weights for each gen')
    parser.add_argument('--plot_param_distrib', default=False, action='store_true', help='plot parameter distribution for each gen')
    parser.add_argument('--plot_kde_matrix', default=False, action='store_true', help='plot KDE matrix of optimization')
    parser.add_argument('--plot_stats', default=False, action='store_true', help='show statisics')
    parser.add_argument('--plot_kdes', default=False, action='store_true', help='plot KDEs of optimzed params')
    parser.add_argument('--plot_fc_sim_vs_data', default=False, action='store_true', help='plot FC of inference from optimization vs real data')
    parser.add_argument('--simulate_posterior_modes', default=False, action='store_true', help='run simulations with at posterior distributions modes ')
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
        kdes,cols = compute_kdes(histories, args=args)
    if args.plot_kdes:
        plot_kdes(kdes, cols, histories=histories, args=args)

    if args.plot_fc_sim_vs_data:
        plot_fc_sim_vs_data(args=args)

    if args.simulate_posterior_modes:
        kde_peaks = get_kde_peaks(kdes)
        sim_outputs = simulate_posterior_modes(kde_peaks, args)
        if args.save_outputs:
            with open(os.path.join(proj_dir, 'postprocessing', 'sims_posterior_modes'+today()+'.pkl'), 'wb') as f:
                pickle.dump(sim_outputs, f)