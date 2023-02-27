# Approximate Bayesian Computation of the Reduced Wong Wang model for OCD
#
# Adapted from https://pyabc.readthedocs.io/en/latest/examples/sde_ion_channels.html
# by Sebastien Naze
#
# QIMR Berghofer 2023

from datetime import datetime
import importlib
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import pyabc
import time

from OCD_modeling.models import ReducedWongWang as RWW
from OCD_modeling.utils.utils import get_working_dir



# model default parameters
default_model_params = dict()
default_model_params['I_0']       = 0.3
default_model_params['J_N']       = 0.2609
default_model_params['w']         = 0.9
default_model_params['G']         = 2.
default_model_params['tau_S']     = 100.
default_model_params['gamma']     = 0.000641
default_model_params['sigma']     = 0.1
default_model_params['N']         = 4
default_model_params['dt']        = 0.01
default_model_params['C']         = np.zeros((default_model_params['N'], default_model_params['N']))

# simulation default parameters
default_sim_params = dict()
default_sim_params['t_tot']         = 1000
default_sim_params['sf']            = 100
default_sim_params['t_rec']         = [100,1000]

# PRIORS
# lower_bound, upper_bound
sigma_min, sigma_max    = 0.01, 0.5     # noise amplitude
G_min, G_max            = 1, 3          # global coupling
C_12_min, C_12_max      = -1, 1         # PFC -> OFC
C_21_min, C_21_max      = -1, 1         # OFC -> PFC
C_13_min, C_13_max      = -1, 1         # NAcc -> OFC
C_31_min, C_31_max      = 0, 1          # OFC -> NAcc
C_24_min, C_24_max      = -1, 1         # Put -> PFC
C_42_min, C_42_max      = 0, 1          # PFC -> Put
C_34_min, C_34_max      = -1, 0         # Put -> NAcc
C_43_min, C_43_max      = -1, 0

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
)

def get_default_args(func):
    """ https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def unpack_params(in_params):
    """ unpack parameters to be given to model and simulation """
    model_params = default_model_params
    #sim_params = get_default_args(RWW.ReducedWongWangND().run)
    sim_params = default_sim_params
    for k,v in in_params.items():
        if k in default_model_params.keys():
            model_params[k] = v
        elif k in default_sim_params.keys():
            sim_params[k] = v
        elif k.startswith('C_'):
                inds = k.split('_')[1]
                j,i = int(inds[0])-1, int(inds[1])-1
                model_params['C'][j,i] = v
        else:
            print('parameter {} is unknown, it is being discarded.'.format(k))
            continue
    return model_params, sim_params

def simulate_rww(params):
    """ instanciate model and simulate trace """
    model_params, sim_params = unpack_params(params)
    #print(model_params)
    rww = RWW.ReducedWongWangND(**model_params)
    rww.run(**sim_params)
    #print(sim_params)
    RWW.compute_bold(rww)
    r, corr_diff, corr = RWW.score_model(rww, coh='pat')
    return {'r': r, 'corr_diff':corr_diff, 'corr':corr}


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


if __name__ == '__main__':
    today = datetime.now().strftime("%Y%m%d")
    trace_name = 'rww4D_'+today
    working_dir = get_working_dir()
    proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_modeling')
    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=32)
    abc = pyabc.ABCSMC(simulate_rww, prior, RWW.distance, sampler=sampler, population_size=1000, max_nr_recorded_particles=1000000)
    abc_id = abc.new(
        "sqlite:///" + os.path.join(proj_dir, 'traces', trace_name+".db"),
        {"r": 1, 'corr_diff':0, 'corr':np.zeros((default_model_params['N'],default_model_params['N']))}  # observation
        # note: observation here is dummy, the distane function does not use it.
    )
    history = abc.run(max_nr_populations=10, minimum_epsilon=0.1)
