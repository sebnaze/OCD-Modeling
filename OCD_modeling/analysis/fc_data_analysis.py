# Module to extend the seed-to-voxel analysis realized in the Brain paper
#
# Author: Sebastien Naze
# Organisation: QIMR Berghofer

import argparse
import importlib
import itertools
import joblib
import nilearn
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
import matplotlib
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import os
import pickle
import platform
import seaborn as sbn
from time import time


# /!\ OCD_baseline imports are not kosher, was quick and dirty way to get things done, regretfully
if platform.node()=='qimr18844':
    import OCD_baseline
    from OCD_baseline.functional.seed_to_voxel_analysis import * 
    import OCD_clinical_trial

from OCD_modeling.utils import get_working_dir, today, emd

working_dir = get_working_dir()
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')
baseline_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCDbaseline')

#-----------------------------#
# -- Baseline reprocessing -- #
#-----------------------------#

def compute_baseline_corr(subjs, rois=['Acc', 'dPut', 'OFC', 'PFC'], args=None):
  """ Computes all correlation between ROIs """
  delayed_funcs = [joblib.delayed(voi_to_voxel)(subj, rois[:-1], metrics=['detrend_gsr_filtered_scrubFD05'], args=args) for subj in subjs]
  joblib.Parallel(n_jobs=args.n_jobs)(delayed_funcs)

def get_subj_rois_corr(subj, sessions=['ses-pre', 'ses-post'], rois=['Acc', 'dPut', 'OFC', 'PFC'], args=None):
    """ create dataframe with correlation between all ROIs  """
    df_lines = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    cohort = get_cohort(subj)
    group = OCD_clinical_trial.functional.get_group(subj)
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for ses in sessions:
            for i,roi_i in enumerate(rois[:-1]):
                # load correlation map
                if ses=='ses-pre':
                    fname = '_'.join([subj, metric, fwhm, atlas, roi_i, 'avg_seed_to_voxel_corr.nii.gz'])
                    fpath = os.path.join(baseline_dir, 'postprocessing', subj, fname)
                elif ses=='ses-post':
                    fname = '_'.join([subj, ses, metric, fwhm, atlas, roi_i, 'avg_seed_to_voxel_corr.nii.gz'])
                    fpath = os.path.join(proj_dir, 'postprocessing', subj, fname)
                else:
                    print("Incorrect session for subj "+subj)
                    break
                if os.path.exists(fpath):
                    corr_map = load_img(fpath)
                else:
                    print(subj+" FC file not found: "+fpath)
                    continue
                for j,roi_j in enumerate(rois[i+1:]):
                    # load voi mask
                    voi_mask = get_voi_mask(roi_j, args)
                    voi_mask = resample_to_img(voi_mask, corr_map, interpolation='nearest')
                    # extract correlations
                    roi_corr = corr_map.get_fdata().copy() * voi_mask.get_fdata().copy()
                    avg_corr = np.mean(roi_corr[roi_corr!=0])
                    df_line = {'subj':subj, 'ses':ses, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 
                               'cohort':cohort, 'group':group, 'pathway':'_'.join([roi_i,roi_j]), 'corr':avg_corr}
                    df_lines.append(df_line)
    return df_lines


def get_rois_corr(subjs, rois=['Acc', 'dPut', 'OFC', 'PFC'], args=None):
    """ parallel post-processing of subjects ROIs correlation  """
    parallel_functions = [joblib.delayed(get_subj_rois_corr)(subj, rois=rois, args=args) for subj in subjs]
    df_lines = joblib.Parallel(n_jobs=args.n_jobs)(parallel_functions)
    df_lines = itertools.chain(*df_lines)
    df_roi_corr = pd.DataFrame(df_lines)

    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_pre-prost_avg_2023.pkl'), 'wb') as f:
            pickle.dump(df_roi_corr, f)

    return df_roi_corr


def plot_rois_corr(df_roi_corr, sessions=['ses-pre', 'ses-post'], rois = ['Acc', 'dPut', 'OFC', 'PFC'], args=None):
    """ (DEPRECATED) boxplots + swarmplots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.6, 0.6]
    for ses in sessions:
        fig = plt.figure(figsize=[3*len(rois), 5*len(rois)])
        gs = plt.GridSpec(len(rois), len(rois))
        for i,roi_i in enumerate(rois[:-1]):
            for j,roi_j in enumerate(rois[i+1:]):
                ax = plt.subplot(gs[i,i+j+1])
                sbn.boxplot(data=df_roi_corr[df_roi_corr['pathway']=='_'.join([roi_i,roi_j])], y='corr', x='pathway', hue='cohort', orient='v', dodge=True, showfliers=False)
                sbn.swarmplot(data=df_roi_corr[df_roi_corr['pathway']=='_'.join([roi_i,roi_j])], y='corr', x='pathway', hue='cohort', orient='v', dodge=True, alpha=0.7, edgecolor='black', linewidth=1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(width=2)
                ax.set_ylim(ylim)
                if i==(len(rois)-1):
                    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                else:
                    ax.get_legend().set_visible(False)
        plt.suptitle(ses)
        plt.tight_layout()

        if args.save_figs:
            figname = 'seeds_to_vois_'+ses+'_avg_FC_'+datetime.now().strftime('_%d%m%Y')+'.svg'
            plt.savefig(os.path.join(proj_dir, 'img', figname))
        plt.show()


def get_subjs(args):
    # if a subject or list of subjects  is given (e.g. if running on HPC), then only process them
    if args.subj!=None:
        subjs = [args.subj]

    elif args.session == 'ses-pre':
        subjs = pd.read_table(os.path.join(code_dir, 'subject_list_all.txt'), names=['name'])['name']
    
    else:
        subjs = OCD_clinical_trial.functional.seed_to_voxel_analysis.get_subjs(args)
        subjs, revoked = OCD_clinical_trial.functional.seed_to_voxel_analysis.get_subjs_after_scrubbing(subjs, 
                                                                                                  seses=['ses-post'], 
                                                                                                  metrics=['detrend_gsr_filtered_scrubFD05'])
    return subjs


#-----------------------------#
# -- Post-TMS reprocessing -- #
#-----------------------------#

def vois_to_voxel_post(subj, vois, metrics, atlases=['Harrison2009'], args=None):
    """ perform voi-to-voxel analysis of bold data post-TMS """
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    ses = 'ses-post'

    for atlas,metric in itertools.product(atlases,metrics):
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        #bold_file = os.path.join(lukeH_deriv_dir, 'post-fmriprep-fix', subj,'func', \
        #                       subj+'_task-rest_space-'+img_space+'_desc-'+metric+'_scrub.nii.gz')
        fname = subj+'_'+ses+'_task-rest_space-'+img_space+'_desc-'+metric+'.nii.gz'
        bold_file = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_clinical_trial/data/derivatives', 
                                    'post-fmriprep-fix', subj, ses, 'func', fname)
        if os.path.exists(bold_file):
            bold_img = nib.load(bold_file)
        else:
            print(subj+' BOLD file not found.')
            continue
        brain_masker = NiftiMasker(smoothing_fwhm=args.brain_smoothing_fwhm, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, standardize='zscore', verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        # extract seed timeseries and perform seed-to-voxel correlation
        for voi in vois:
            voi_img = OCD_baseline.functional.seed_to_voxel_analysis.get_voi_mask(voi, args)
            voi_masker = NiftiMasker(voi_img, t_r=0.81, low_pass=0.1, high_pass=0.01, verbose=0)
            voi_ts = np.mean(voi_masker.fit_transform(bold_img), axis=1)
            voi_ts = (voi_ts - np.mean(voi_ts))/np.std(voi_ts)
            voi_to_voxel_corr = np.dot(voxels_ts.T, voi_ts)/voxels_ts.shape[0]
            voi_to_voxel_corr_img = brain_masker.inverse_transform(voi_to_voxel_corr.squeeze())
            fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
            fname = '_'.join([subj, ses, metric,fwhm,atlas,voi,'avg_seed_to_voxel_corr.nii.gz'])
            nib.save(voi_to_voxel_corr_img, os.path.join(out_dir, fname))
    print('{} voi_to_voxel correlation performed in {}s'.format(subj,int(time()-t0)))


def compute_post_corr(rois=['Acc', 'dPut', 'OFC', 'PFC'], args=None):
  """ Computes correlation between ROIs """
  subjs = OCD_clinical_trial.functional.seed_to_voxel_analysis.get_subjs(args)
  subjs, revoked = OCD_clinical_trial.functional.seed_to_voxel_analysis.get_subjs_after_scrubbing(subjs, 
                                                                                                  seses=['ses-post'], 
                                                                                                  metrics=['detrend_gsr_filtered_scrubFD05'])
  delayed_funcs = [joblib.delayed(vois_to_voxel_post)(subj, rois[:-1], metrics=['detrend_gsr_filtered_scrubFD05'], args=args) for subj in subjs]
  joblib.Parallel(n_jobs=args.n_jobs)(delayed_funcs)


def load_rois_corr(args):
    """ Load precomputed correlation maps between ROIs """
    fname = os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_pre-post_avg_2023.pkl')
    with open(fname, 'rb') as f:
        df_rois_corr = pickle.load(f)
    return df_rois_corr


def prep_pre_post_df(df_rois_corr):
    """ Add a ses-pre / ses-post for controls and fix None groups """
    # add ses-post for controls
    df_controls = df_rois_corr[df_rois_corr.cohort=='controls']
    df_controls['ses'] = 'ses-post'
    df_rois_corr = pd.concat([df_rois_corr, df_controls]).sort_values('subj')

    # remove patients with no group attributed
    df_rois_corr.drop(index=df_rois_corr[(df_rois_corr.cohort=='patients') & (df_rois_corr.group=='none')].index, inplace=True)
    return df_rois_corr


def plot_pre_post_fc_vs_controls(df_rois_corr, args):
    """ Box plots of controls vs active vs sham FC in pathways """
    pathways = df_rois_corr.pathway.unique()
    pathways.sort()

    fig = plt.figure(figsize=[12,3])
    ax = plt.subplot(1,2,1)
    sbn.boxplot(data=df_rois_corr[df_rois_corr.ses=='ses-pre'], x='pathway', y='corr', hue='group', ax=ax, order=pathways)
    ax.legend().set_visible(False)
    plt.xticks(rotation=30)
    plt.ylim([-0.5,0.5])
    plt.title('baseline')

    ax = plt.subplot(1,2,2)
    sbn.boxplot(data=df_rois_corr[df_rois_corr.ses=='ses-post'], x='pathway', y='corr', hue='group', ax=ax, order=pathways)
    sbn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=30)
    plt.ylim([-0.5,0.5])
    plt.title('post TMS')

#----------------------------------#
# Distance analysis to FC controls
#----------------------------------#

def compute_distances(df_rois_corr):
    """ Compute Wassertein and euclidian distances between distributions across groups """ 
    pathways = np.sort(df_rois_corr.pathway.unique())
    distances = {'indiv':pd.DataFrame([]), 'group':[]}
    for ses,group in itertools.product(['ses-pre', 'ses-post'], ['group1', 'group2']):
        df_con = df_rois_corr[(df_rois_corr.cohort=='controls') & (df_rois_corr.ses==ses)].pivot(
            index='subj', columns='pathway', values='corr').reset_index()
        df_pat = df_rois_corr[(df_rois_corr.group==group) & (df_rois_corr.ses==ses)].pivot(
            index='subj', columns='pathway', values='corr').reset_index()
        
        # compute the wasserstein distance between groups
        d = emd(df_con[pathways], df_pat[pathways])
        distances['group'].append({'ses':ses, 'group': group, 'distance': d})

        # compute euclidian distances to mean controls
        ref = df_con[pathways].apply(np.mean, axis=0)
        def dist(x):
            return np.sqrt(np.sum(np.array(x)-np.array(ref)**2))
        
        # format outputs into dataframe
        df_con['dist'] = df_con[pathways].apply(dist, axis=1)
        df_con['ses'] = ses
        df_con['cohort'] = 'controls'
        df_con['group'] = 'none'
        df_pat['dist'] = df_pat[pathways].apply(dist, axis=1)
        df_pat['ses'] = ses
        df_pat['cohort'] = 'patients'
        df_pat['group'] = group
        distances['indiv'] = pd.concat([distances['indiv'], df_con, df_pat])
    distances['group'] = pd.DataFrame(distances['group'])
    distances['indiv'].drop_duplicates(inplace=True, ignore_index=True)
    return distances


def plot_pointplot_pre_post(df):
    """ Point plot of individuals pre vs post distance to controls FC based on euclidean distance"""
    plt.figure(figsize=[12,4])

    plt.subplot(1,3,1)
    df_tmp = df[df.group=='none']
    for subj in df_tmp.subj.unique():
        sbn.pointplot(data=df_tmp[df_tmp.subj==subj], x='ses', y='dist', color='lightblue', lw=0.2)
    plt.title('controls')
    plt.ylim([0,1.2])
    plt.gca().spines.top.set_visible(False)
    plt.gca().spines.right.set_visible(False)

    plt.subplot(1,3,2)
    df_tmp = df[df.group=='group1']
    responders = {'pre':[], 'post':[]}
    for subj in df_tmp.subj.unique():
        pre = df_tmp[(df_tmp.subj==subj) & (df_tmp.ses=='ses-pre')]['dist'].iloc[0]
        post = df_tmp[(df_tmp.subj==subj) & (df_tmp.ses=='ses-post')]['dist'].iloc[0]
        if pre > post:
            sbn.pointplot(data=df_tmp[df_tmp.subj==subj], x='ses', y='dist', order=['ses-pre', 'ses-post'], color='orange', lw=0.2)
            responders['pre'].append(pre)
            responders['post'].append(post)
        else:
            sbn.pointplot(data=df_tmp[df_tmp.subj==subj], x='ses', y='dist', order=['ses-pre', 'ses-post'], color='gold', lw=0.2, alpha=0.4)
    val_1 = df[(df.group=='group1') & (df.ses=='ses-pre')].dist
    val_2 = df[(df.group=='group1') & (df.ses=='ses-post')].dist
    t,p = scipy.stats.ttest_rel(val_1, val_2)
    tr,pr = scipy.stats.ttest_rel(responders['pre'], responders['post'])
    plt.title('group1\nn={}  t={:.2f}  p={:.3f}\nResponders: n={}  t={:.2f}  p={:.3f}'.format(
        len(val_1), t, p, len(responders['pre']), tr, pr))
    plt.ylim([0,1.2])
    plt.gca().spines.top.set_visible(False)
    plt.gca().spines.right.set_visible(False)

    plt.subplot(1,3,3)
    df_tmp = df[df.group=='group2']
    responders = {'pre':[], 'post':[]}
    for subj in df_tmp.subj.unique():
        pre = df_tmp[(df_tmp.subj==subj) & (df_tmp.ses=='ses-pre')]['dist'].iloc[0]
        post = df_tmp[(df_tmp.subj==subj) & (df_tmp.ses=='ses-post')]['dist'].iloc[0]
        if pre > post:
            responders['pre'].append(pre)
            responders['post'].append(post)
            sbn.pointplot(data=df_tmp[df_tmp.subj==subj], x='ses', y='dist', order=['ses-pre', 'ses-post'], color='green', lw=0.2)
        else:
            sbn.pointplot(data=df_tmp[df_tmp.subj==subj], x='ses', y='dist', order=['ses-pre', 'ses-post'], color='lime', lw=0.2, alpha=0.4)
    val_1 = df[(df.group=='group2') & (df.ses=='ses-pre')].dist
    val_2 = df[(df.group=='group2') & (df.ses=='ses-post')].dist
    t,p = scipy.stats.ttest_rel(val_1, val_2)
    tr,pr = scipy.stats.ttest_rel(responders['pre'], responders['post'])
    plt.title('group2\nn={}  t={:.2f}  p={:.3f}\nResponders: n={}  t={:.2f}  p={:.3f}'.format(
        len(val_1), t, p, len(responders['pre']), tr, pr))
    plt.ylim([0,1.2])
    plt.gca().spines.top.set_visible(False)
    plt.gca().spines.right.set_visible(False)

    plt.tight_layout()
    plt.show()


def drop_single_session(df):
    """ remove subjects with only pre or only post sessions """
    rm_subjA = [s for s in df[df.ses=='ses-pre'].subj.to_list() if not s in df[df.ses=='ses-post'].subj.to_list()]
    rm_subjB = [s for s in df[df.ses=='ses-post'].subj.to_list() if not s in df[df.ses=='ses-pre'].subj.to_list()]
    rm_subj = np.unique(np.concatenate([rm_subjA, rm_subjB]))
    for subj in rm_subj:
        df.drop(index=df[df.subj==subj].index, inplace=True)
    df.sort_values('subj', inplace=True, ignore_index=True)
    return df


def main(args):
    subjs = get_subjs(args)    
    if args.compute_baseline_roi_corr:
        compute_baseline_corr(subjs, args=args)

    if args.compute_post_tms_roi_corr:
        compute_post_corr(args=args)

    if args.load_rois_corr:
        df_rois_corr = load_rois_corr(args)
    else:
        df_rois_corr = get_rois_corr(subjs, args=args)
                                                                                  
    df_rois_corr = prep_pre_post_df(df_rois_corr)
    if args.plot_pre_post_fc_vs_controls:
        plot_pre_post_fc_vs_controls(df_rois_corr, args)

    if args.compute_distances:
        distances = compute_distances(df_rois_corr)

    df_fc_dist = drop_single_session(distances['indiv'])
    if args.plot_pointplot_pre_post:
        plot_pointplot_pre_post(df_fc_dist)

    if args.save_outputs:
        fname= os.path.join(proj_dir, 'postprocessing', 'distances_to_FC_controls'+today()+'.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(distances, f)

    return df_rois_corr, distances

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_baseline_roi_corr', default=False, action="store_true", help="compute correlation between vois, seeds, in all directions")
    parser.add_argument('--compute_post_tms_roi_corr', default=False, action="store_true", help="compute correlation between vois, seeds, in all directions")
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--seed_type', default='Harrison2009', type=str, action='store', help='choose Harrison2009, TianS4, etc')
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--brain_smoothing_fwhm', default=8., type=none_or_float, action='store', help='brain smoothing FWHM (default 8mm as in Harrison 2009)')
    parser.add_argument('--subj', default=None, action='store', help='to process a single subject, give subject ID (default: process all subjects)')
    parser.add_argument('--unilateral_seed', default=False, action='store_true', help='use unilateral seed (Acc, dPut)')
    parser.add_argument('--session', default='ses-pre', action='store', help='which session to recompute FC (ses-pre (default) or ses-post)')
    parser.add_argument('--load_rois_corr', default=False, action='store_true', help='load precomputed correlations between ROIs, if not set, compute them')
    parser.add_argument('--plot_pre_post_fc_vs_controls', default=False, action='store_true', help='')
    parser.add_argument('--compute_distances', default=False, action='store_true', help='compute the wasserstein distances to controls in pre and post intervention')
    parser.add_argument('--plot_pointplot_pre_post', default=False, action='store_true', help='pointplot of FC difference between pre and post based on distacne to avg controls FC')
    args = parser.parse_args()
    # hard coded
    args.metrics = ['detrend_gsr_filtered_scrubFD05']
    args.atlases = ['Harrison2009']
    args.deriv_mask = 'Shephard'
    return args


if __name__=='__main__':
  args = parse_arguments()
  df_rois_corr, distances = main(args)
