# Launcher of the OCD model on hpc
#
# Author: Sebastien Naze
#
# QIMR Berghofer 2022-2023

import argparse
import json
import numpy as np
import os
import pickle
import platform
import sys
import time

import OCD_modeling
from OCD_modeling.models import ReducedWongWang as RWW

# get computer name to set paths
if platform.node()=='qimr18844':
    working_dir = '/home/sebastin/working/'
elif 'hpcnode' in platform.node():
    working_dir = '/mnt/lustre/working/'
else:
    print('Computer unknown! Setting working dir as /working')
    working_dir = '/working/'

# general paths
proj_dir = working_dir+'lab_lucac/sebastiN/projects/OCD_modeling'

def load_params(args):
    """ load parameters set based on input index """
    with open(os.path.join(proj_dir, 'log', 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    return params[args.params_idx]

def main(args):
    print("Batch {}".format(args.batch_id))
    t0 = time.time()
    p = load_params(args)
    C = np.array([  [0,p['C_12'],p['C_13'],0], # to OFC
                    [p['C_21'],0,0,p['C_24']], # to PFC
                    [p['C_31'],0,0,p['C_34']], # to NAcc
                    [0,p['C_42'],p['C_43'],0]  # to Put
                ])
    rww = RWW.ReducedWongWang2D(dt=0.01, N=4, C=C, G=p['G'], sigma=p['sigma'])
    print("Running model #{:08d}".format(args.params_idx))
    rww.run(t_tot=5000, sf=100)
    score = RWW.score_model(rww, t_range=[2000, 5000])

    if args.save_scores:
        print("Saving score for model #{:08d}".format(args.params_idx))
        fpath = os.path.join(proj_dir, 'log', args.batch_id, 'paramID_{:08d}'.format(args.params_idx))
        os.makedirs(fpath, exist_ok=True)
        with open(os.path.join(fpath, 'score.pkl'), 'wb') as f:
            pickle.dump(score, f)
        with open(os.path.join(fpath, 'param.txt'), 'w') as f:
            f.write(json.dumps(p))
    print("Done in {:.2f}.".format(time.time()-t0))

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
  parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
  parser.add_argument('--save_scores', default=False, action='store_true', help='save scores')
  parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
  parser.add_argument('--params_idx', type=int, default=0, action='store', help="index of the parameter set")
  parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
  parser.add_argument('--batch_id', type=str, default="test", action='store', help='batch number unique to each batched job launched on cluster, typically YYYYMMDDD_hhmmss')
  args = parser.parse_args()
  return args

if __name__=='__main__':
  args = parse_arguments()
  main(args)
