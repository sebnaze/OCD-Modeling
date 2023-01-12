# Module to extend the seed-to-voxel analysis realized in the Brain paper
#
# Author: Sebastien Naze
# Organisation: QIMR Berghofer

import importlib
import itertools
import joblib
import functional
import functional.seed_to_voxel_analysis
importlib.reload(functional.seed_to_voxel_analysis)
from functional.seed_to_voxel_analysis import *

proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')
baseline_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCDbaseline')

def compute_all_corr(subjs, rois=['Acc', 'dPut', 'OFC', 'PFC'], args=None):
  """ Computes all correlation between ROIs """
  delayed_funcs = [joblib.delayed(voi_to_voxel)(subj, rois[:-1], metrics=['detrend_gsr_filtered_scrubFD05'], args=args) for subj in subjs]
  joblib.Parallel(n_jobs=args.n_jobs)(delayed_funcs)

def get_subj_rois_corr(subj, rois=['Acc', 'dPut', 'OFC', 'PFC'], args=None):
    """ create dataframe with correlation between all ROIs  """
    df_lines = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    cohort = get_cohort(subj)
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for i,roi_i in enumerate(rois[:-1]):
            # load correlation map
            fname = '_'.join([subj, metric, fwhm, atlas, roi_i, 'avg_seed_to_voxel_corr.nii.gz'])
            fpath = os.path.join(baseline_dir, 'postprocessing', subj, fname)
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
                df_line = {'subj':subj, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'cohort':cohort, 'pathway':'_'.join([roi_i,roi_j]), 'corr':avg_corr}
                df_lines.append(df_line)
    return df_lines


def get_rois_corr(subjs, rois=['Acc', 'dPut', 'OFC', 'PFC'], args=None):
    """ parallel post-processing of subjects ROIs correlation  """
    parallel_functions = [joblib.delayed(get_subj_rois_corr)(subj, rois=rois, args=args) for subj in subjs]
    df_lines = joblib.Parallel(n_jobs=args.n_jobs)(parallel_functions)
    df_lines = itertools.chain(*df_lines)
    df_roi_corr = pd.DataFrame(df_lines)

    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_avg.pkl'), 'wb') as f:
            pickle.dump(df_roi_corr, f)

    return df_roi_corr


def plot_rois_corr(df_roi_corr, rois = ['Acc', 'dPut', 'OFC', 'PFC'], args=None):
    """ violinplots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.6, 0.6]
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
    plt.tight_layout()

    if args.save_figs:
        figname = 'seeds_to_vois_avg_FC_'+datetime.now().strftime('_%d%m%Y')+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', figname))

def get_subjs(args):
  # if a subject or list of subjects  is given (e.g. if running on HCP), then only process them
  if args.subj!=None:
      subjs = [args.subj]
  # otherwise process all subjects
  else:
      subjs = pd.read_table(os.path.join(code_dir, 'subject_list_all.txt'), names=['name'])['name']
  return subjs



def main(args):
    subjs = get_subjs(args)
    if args.compute_roi_corr:
        compute_all_corr(subjs, args=args)
    if args.plot_figs:
        df_roi_corr = get_rois_corr(subjs, args=args)
        plot_rois_corr(df_roi_corr, args=args)



def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--compute_roi_corr', default=False, action="store_true", help="compute correlation between vois, seeds, in all directions")
  parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
  parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
  parser.add_argument('--seed_type', default='Harrison2009', type=str, action='store', help='choose Harrison2009, TianS4, etc')
  parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
  parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
  parser.add_argument('--brain_smoothing_fwhm', default=8., type=none_or_float, action='store', help='brain smoothing FWHM (default 8mm as in Harrison 2009)')
  parser.add_argument('--subj', default=None, action='store', help='to process a single subject, give subject ID (default: process all subjects)')
  args = parser.parse_args()
  # hard coded
  args.metrics = ['detrend_gsr_filtered_scrubFD05']
  args.atlases = ['Harrison2009']
  args.deriv_mask = 'Shephard'
  return args


if __name__=='__main__':
  args = parse_arguments()
  main(args)
