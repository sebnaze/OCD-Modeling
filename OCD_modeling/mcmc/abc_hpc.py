# Approximate Bayesian Computation of the Reduced Wong Wang model for OCD
#
# Adapted from https://pyabc.readthedocs.io/en/latest/examples/sde_ion_channels.html
# by Sebastien Naze
#
# QIMR Berghofer 2023

import argparse
from datetime import datetime
import importlib
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import pyabc
import time

import OCD_modeling
from OCD_modeling.models import ReducedWongWang as RWW
from OCD_modeling.hpc import parallel_launcher
from OCD_modeling.utils import utils

working_dir = utils.get_working_dir()
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')


# model default parameters
default_model_params = dict()
default_model_params['I_0']       = 0.3
default_model_params['J_N']       = 0.2609
default_model_params['w']         = 0.9
default_model_params['G']         = 2.5
default_model_params['tau_S']     = 100.
default_model_params['gamma']     = 0.000641
default_model_params['sigma']     = 0.1
default_model_params['N']         = 4
default_model_params['dt']        = 0.01
default_model_params['C']         = np.zeros((default_model_params['N'], default_model_params['N']))
default_model_params['sigma_C']   = np.zeros((default_model_params['N'], default_model_params['N']))
default_model_params['eta_C']     = np.zeros((default_model_params['N'], default_model_params['N']))

# simulation default parameters
default_sim_params = dict()
default_sim_params['t_tot']         = 8000
default_sim_params['rec_vars']      = ['C_13', 'C_24']
default_sim_params['sf']            = 100

# BOLD default parameters
default_bold_params = dict()
default_bold_params['t_range']      = [5000,8000]
default_bold_params['transient']    = 60

# PRIORS
# lower_bound, upper_bound
sigma_min, sigma_max    = 0.05, 0.1     # noise amplitude
G_min, G_max            = 2, 3           # global coupling
C_12_min, C_12_max      = 0, 0.5        # PFC -> OFC
C_21_min, C_21_max      = 0, 0.5        # OFC -> PFC
C_13_min, C_13_max      = -0.1, 0.5      # NAcc -> OFC
C_31_min, C_31_max      = 0, 0.5         # OFC -> NAcc
C_24_min, C_24_max      = -0.1, 0.5      # Put -> PFC
C_42_min, C_42_max      = 0, 0.5         # PFC -> Put
C_34_min, C_34_max      = -0.5, 0.5      # Put -> NAcc
C_43_min, C_43_max      = -0.5, 0.5      # Nacc -> Put

# coupling noises on Ornstein-Uhlenbeck process 
sigma_C_13_min, sigma_C_13_max  = 0.1, 0.4
sigma_C_24_min, sigma_C_24_max  = 0.1, 0.4
eta_C_13_min, eta_C_13_max  = 0, 0.1
eta_C_24_min, eta_C_24_max  = 0, 0.1


prior = pyabc.Distribution(
    sigma = pyabc.RV("uniform", sigma_min, sigma_max - sigma_min),
    G = pyabc.RV("uniform", G_min, G_max - G_min),
    C_12 = pyabc.RV("uniform", C_12_min, C_12_max - C_12_min),
    C_21 = pyabc.RV("uniform", C_21_min, C_21_max - C_21_min),
    C_13 = pyabc.RV("uniform", C_13_min, C_13_max - C_13_min),
    C_31 = pyabc.RV("uniform", C_31_min, C_31_max - C_31_min),
    C_24 = pyabc.RV("uniform", C_24_min, C_24_max - C_24_min),
    C_42 = pyabc.RV("uniform", C_42_min, C_42_max - C_42_min),
    C_34 = pyabc.RV("uniform", C_34_min, C_34_max - C_34_min),
    C_43 = pyabc.RV("uniform", C_43_min, C_43_max - C_43_min),
    sigma_C_13 = pyabc.RV("uniform", sigma_C_13_min, sigma_C_13_max - sigma_C_13_min),
    sigma_C_24 = pyabc.RV("uniform", sigma_C_24_min, sigma_C_24_max - sigma_C_24_min),
    eta_C_13 = pyabc.RV("uniform", eta_C_13_min, eta_C_13_max - eta_C_13_min),
    eta_C_24 = pyabc.RV("uniform", eta_C_24_min, eta_C_24_max - eta_C_24_min)
)

def get_default_args(func):
    """ https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value """
    """ (actually not in use -- deprecated) """ 
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def unpack_params(in_params):
    """ unpack parameters to be given to model and simulation """
    model_params = default_model_params
    sim_params = default_sim_params
    bold_params = default_bold_params
    control_params = dict()  # if keept empty, unused 

    # unpack model's params
    for k,v in in_params.items():
        if k in default_model_params.keys():
            model_params[k] = v
        elif ('C_' in k):
            var, inds = k.rsplit('_', maxsplit=1)
            j,i = int(inds[0])-1, int(inds[1])-1
            model_params[var][j,i] = v

        # unpack sim params
        elif k in default_sim_params.keys():
            sim_params[k] = v
        
        # unpack bold params
        elif k in default_bold_params.keys():
            bold_params[k] = v
        else:
            print('parameter {} is unknown, it is being discarded.'.format(k))
            continue
    return model_params, sim_params, control_params, bold_params

def simulate_rww(params):
    """ instanciate model and simulate trace """
    model_params, sim_params, control_params, bold_params = unpack_params(params)
    #print(model_params)
    rww = RWW.ReducedWongWangND(**model_params)
    rww.run(**sim_params)
    #print(sim_params)
    RWW.compute_bold(rww)
    r, corr_diff, corr = RWW.score_model(rww, coh='pat')
    return {'r': r, 'corr_diff':corr_diff, 'corr':corr}


def simulate_population_rww(params):
    """ instanciate models and simulate traces """
    args = argparse.Namespace()
    args.n_sims = 56
    args.n_jobs = 8
    args.model_pars, args.sim_pars, args.control_pars, args.bold_pars = unpack_params(params)
    
    #sim_objs = parallel_launcher.launch_simulations(args)
    sim_objs = parallel_launcher.launch_pool_simulations(args)
    
    RMSE = RWW.score_population_models(sim_objs, cohort='controls')
    return {'RMSE': RMSE}



def evaluate_prediction(history, n_samples=1000):
    """ Re-run model for the best scored parameters (highest posteriors) """
    # get paramaters from best estimates and re-simulate with it
    pop_ext = history.get_population_extended()
    cols = pop_ext.columns[2:-1]
    res=[]
    for i in range(n_samples):
        res.append(dict((k[4:],v) for k,v in pop_ext.iloc[i][cols].items()))

    output = joblib.Parallel(n_jobs=28)(joblib.delayed(simulate_rww)(param) for param in res)
    return output


def sim_output_to_df(sim_output, coh='con'):
    """ reformats simluation output to dataframe """ 
    df = [] 
    for i,out in enumerate(sim_output):
        df.append({'subj':'sim-{}{:04d}'.format(coh,i), 'pathway':'OFC_PFC', 'cohort':'sim-'+coh, 'corr':out['corr'][0,1]})
        df.append({'subj':'sim-{}{:04d}'.format(coh,i), 'pathway':'Acc_OFC', 'cohort':'sim-'+coh, 'corr':out['corr'][0,2]})
        df.append({'subj':'sim-{}{:04d}'.format(coh,i), 'pathway':'dPut_OFC', 'cohort':'sim-'+coh, 'corr':out['corr'][0,3]})
        df.append({'subj':'sim-{}{:04d}'.format(coh,i), 'pathway':'Acc_PFC', 'cohort':'sim-'+coh, 'corr':out['corr'][1,2]})
        df.append({'subj':'sim-{}{:04d}'.format(coh,i), 'pathway':'dPut_PFC', 'cohort':'sim-'+coh, 'corr':out['corr'][1,3]})
        df.append({'subj':'sim-{}{:04d}'.format(coh,i), 'pathway':'Acc_dPut', 'cohort':'sim-'+coh, 'corr':out['corr'][2,3]})
    df_sim = pd.DataFrame(df)
    return df_sim


def get_config():
    """ extract redis config """
    cfg = dict()
    with open(os.path.join(proj_dir, 'envs', 'redis.conf'), 'r') as f:
        line = f.readline()
        cfg['pssr'] = line.split(' ')[1].strip()
    with open(os.path.join(proj_dir, 'traces', '.redis_ip'), 'r') as f:
        cfg['redis_ip'] = f.readline().strip()
    return cfg


def parse_args():
    """ Parse command line arguments """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--redis_ip', type=str, default="10.10.58.254", action='store', help='set ip address of the server. default: hpcapp01')
    parser.add_argument('--redis_pwd', type=str, default='bayesopt1234321', action='store', help='password to access server')
    parser.add_argument('--resume_opt', type=str, default=None, action='store', help='name of the optimzation to resume (default:None, start a new optimization)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    today = datetime.now().strftime("%Y%m%d")
    trace_name = 'rww4D_OU_HPC_'+today
    working_dir = OCD_modeling.utils.utils.get_working_dir() 
    proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')
    #sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=1)
    sampler = pyabc.sampler.RedisEvalParallelSampler(host=cfg['redis_ip'], port=6379, password=cfg['pssr'])
    sampler.daemon = False
    abc = pyabc.ABCSMC(simulate_population_rww, prior, RWW.distance, sampler=sampler, population_size=1000, max_nr_recorded_particles=20000)
    if args.resume_opt==None:
        abc_id = abc.new(
            "sqlite:///" + os.path.join(proj_dir, 'traces', trace_name+".db"),
            {"RMSE": 0}  # observation # note: here is dummy, the distance function does not use it.
        )
    else:
        abc.load("sqlite:///" + os.path.join(proj_dir, 'traces', args.resume_opt+".db"), abc_id=1)
    history = abc.run(max_nr_populations=10, minimum_epsilon=0.01)
