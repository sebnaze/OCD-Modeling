# Approximate Bayesian Computation of the Reduced Wong Wang model for OCD
#
# Adapted from https://pyabc.readthedocs.io/en/latest/examples/sde_ion_channels.html
# by Sebastien Naze
#
# QIMR Berghofer 2023

import argparse
from datetime import datetime
import itertools
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import pyabc
import time
import tomli

import OCD_modeling
from OCD_modeling.models import ReducedWongWang as RWW
from OCD_modeling.hpc import parallel_launcher
from OCD_modeling.utils.utils import working_dir, proj_dir, today, read_config

default_params = dict()


def set_default_params(config):
    """ Set default parameters from the config file """
    default_params['model'] = config['default_params']['model']
    default_params['sim'] = config['default_params']['sim']
    default_params['bold'] = config['default_params']['bold']
    default_params['control'] = dict()
    default_params['cohort'] = config['optim_params']['cohort']

def get_config_priors(config):
    """ Creates priors objects from TOML config file. 
    
    Parameters
    ----------
        config: dict
            TOML config file of the project.
    
    Returns
    -------
        priors: dict
            Prior distribution for each parameters.
        bounds: dict
            Min and Max values of the parameters.

    """

    conf_priors = config['optim_params']['priors']
    
    # outputs
    priors, bounds = dict(), dict()
    
    for k,v in conf_priors.items():
        if 'C_' in k:
            var = k.rsplit(sep='_', maxsplit=2)[0]
            vals = np.array(v)
            x,y,_ = vals.shape
            for i,j in itertools.product(np.arange(x), np.arange(y)):
                if (vals[i,j,1] - vals[i,j,0])==0: continue  # case of same value, e.g. [0,0], discard 
                par = var+'_'+str(i+1)+str(j+1)
                priors[par] = pyabc.RV("uniform", vals[i,j,0], vals[i,j,1] - vals[i,j,0])
                bounds[par] = vals[i,j]
        else:
            if (v[1]-v[0])==0: continue
            bounds[k] = v
            priors[k] = pyabc.RV("uniform", v[0], v[1] - v[0])

    return pyabc.Distribution(priors), bounds


def unpack_params(in_params):
    """ Unpack parameters to be given to model and simulation.
    
    All type of input parameter can be given, it will be unpacked and atrributed to the correct parameter dictionnary.
    If the parameter is not recognized, a warning will be displayed and the input parameters will be ignored.

    Parameters
    ----------
        in_params: dict
            Input parameters. 

    Returns
    -------
        model_params: dict
            Model parameters.
        sim_params: dict
            Simulations parameters.
        control_params: dict
            Control parameters.
        bold_params: dict
            BOLD parameters.
    """
    #model_params, sim_params, control_params, bold_params = get_default_params()
    model_params = default_params['model'].copy()
    sim_params = default_params['sim'].copy()
    control_params = default_params['control'].copy()
    bold_params = default_params['bold'].copy()
    cohort = default_params['cohort']

    # unpack model's params
    for k,v in in_params.items():
        if k in default_model_params.keys():
            model_params[k] = v
        elif ('C_' in k):
            var, inds = k.rsplit('_', maxsplit=1)
            j,i = int(inds[0])-1, int(inds[1])-1
            model_params[var] = np.array(model_params[var])
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
    return model_params, sim_params, control_params, bold_params, cohort

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
    args.n_sims = 8
    args.n_jobs = 8
    args.model_pars, args.sim_pars, args.control_pars, args.bold_pars, cohort = unpack_params(params)
    
    #sim_objs = parallel_launcher.launch_simulations(args)
    sim_objs = parallel_launcher.launch_pool_simulations(args)
    
    RMSE = RWW.score_population_models(sim_objs, cohort=cohort)
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


def parse_args():
    """ Parse command line arguments """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType('rb'), help='Project config file to use (TOML file).')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    set_default_params(config)
    sampler = pyabc.sampler.RedisEvalParallelSampler(host=config['optim_params']['ip_address'], 
                                                     port=6379, 
                                                     password=config['optim_params']['password'])
    #sampler = pyabc.sampler.RedisEvalParallelSampler(host="10.10.58.254", port=6379, password='bayesopt1234321')
    #sampler = pyabc.sampler.RedisEvalParallelSampler(host="10.1.121.227", port=6379, password='bayesopt1234321')
    sampler.daemon = False
    prior, _ = get_config_priors(config)
    abc = pyabc.ABCSMC(simulate_population_rww, prior, RWW.distance, sampler=sampler, 
                       population_size=config['optim_params']['population_size'])

    os.makedirs(os.path.join(proj_dir, 'traces'), exist_ok=True)
    abc_id = abc.new(
        "sqlite:///" + os.path.join(proj_dir, 'traces', config['optim_params']['db_name']+".db"),
        {"RMSE": 0}  # observation # note: here is dummy, the distance function does not use it.
    )
    history = abc.run(max_nr_populations=config['optim_params']['max_nr_populations'], 
                      minimum_epsilon=config['optim_params']['min_epsilon'])
