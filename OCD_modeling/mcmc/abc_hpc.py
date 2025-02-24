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

#import OCD_modeling
from OCD_modeling.models import ReducedWongWang as RWW
from OCD_modeling.hpc import parallel_launcher
from OCD_modeling.utils.utils import *


"""
/!\ IMPORTANT /!\

To optimize against OCD or healthy controls, you must change the cohort parameter in simulate_population_rww function:
RMSE = RWW.score_population_models(sim_objs, cohort='controls')
#RMSE = RWW.score_population_models(sim_objs, cohort='patients')

When changing the number of nodes/regions of the simulation, these edits must be done:
    def get_default_params(N=4)  <-- first function below, default to N=4, must be changed to N wanted

    In main, the priors must be set by hand (choose between get_prior, get_prior_Thal, get_prior_Thal_fc_weak, etc.) via comment/uncomment
    or create you own prior function if your N is not currently supported (i.e. N=4 or N=6)

    prior, _ = get_prior()
    #prior, _ = get_prior_Thal()  # <-- hc_strong
    #prior, _ = get_prior_Thal_hc_weak()
    #prior, _ = get_prior_Thal_fc_weak()

"""



def get_default_params(N=4):
    """ Create a structure with default models, simulation and BOLD parameters

    Parameters
    ----------
        N: int
            Number of regions to model

    Returns
    -------
        default_model_params: dict
            Model parameters.
        default_sim_params: dict
            Simulation parameters.
        default_control_params: dict
            Control parameters.
        default_bold_params: dict
            BOLD parameters.
    """

    # model default parameters
    default_model_params = dict()
    default_model_params['I_0']       = 0.3
    default_model_params['J_N']       = 0.2609
    default_model_params['w']         = 0.9
    default_model_params['G']         = 2.5
    default_model_params['tau_S']     = 100.
    default_model_params['gamma']     = 0.000641
    default_model_params['sigma']     = 0.1
    default_model_params['N']         = N
    default_model_params['dt']        = 0.01
    default_model_params['C']         = np.zeros((default_model_params['N'], default_model_params['N']))
    default_model_params['sigma_C']   = np.zeros((default_model_params['N'], default_model_params['N']))
    default_model_params['eta_C']     = np.zeros((default_model_params['N'], default_model_params['N']))

    # simulation default parameters
    default_sim_params = dict()
    default_sim_params['t_tot']         = 8000
    default_sim_params['rec_vars']      = []#['C_13', 'C_24']
    default_sim_params['sf']            = 100

    # Control default parameters
    default_control_params = dict()

    # BOLD default parameters
    default_bold_params = dict()
    default_bold_params['t_range']      = [5000,8000]
    default_bold_params['transient']    = 60

    return default_model_params, default_sim_params, default_control_params, default_bold_params


def get_prior():
    """ Create uniform prior distributions of parameters to start Sequential Monte-Carlo.
    
    Returns
    -------
        prior: pyABC.Distribution
            Distribution object of priors.
        bounds: dict
            Lower and upper bounds of parameters , i.e. bounds['param_name']=[min,max].
    """

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
        eta_C_24 = pyabc.RV("uniform", eta_C_24_min, eta_C_24_max - eta_C_24_min) )

    bounds = dict()
    bounds['sigma'] = [sigma_min, sigma_max]
    bounds['G'] = [G_min, G_max]
    bounds['C_12'] = [C_12_min, C_12_max]
    bounds['C_21'] = [C_21_min, C_21_max]
    bounds['C_13'] = [C_13_min, C_13_max]
    bounds['C_31'] = [C_31_min, C_31_max]
    bounds['C_24'] = [C_24_min, C_24_max]
    bounds['C_42'] = [C_42_min, C_42_max]
    bounds['C_34'] = [C_34_min, C_34_max]
    bounds['C_43'] = [C_43_min, C_43_max]
    bounds['eta_C_13'] = [eta_C_13_min, eta_C_13_max]
    bounds['sigma_C_13'] = [sigma_C_13_min, sigma_C_13_max]
    bounds['eta_C_24'] = [eta_C_24_min, eta_C_24_max]
    bounds['sigma_C_24'] = [sigma_C_24_min, sigma_C_24_max]
    return prior, bounds


def get_prior_Thal():
    """ Create uniform prior distributions of parameters to start Sequential Monte-Carlo.
    
    Returns
    -------
        prior: pyABC.Distribution
            Distribution object of priors.
        bounds: dict
            Lower and upper bounds of parameters , i.e. bounds['param_name']=[min,max].
    """

    # PRIORS
    # lower_bound, upper_bound
    sigma_min, sigma_max    = 0.05, 0.1      # noise amplitude
    G_min, G_max            = 2, 3           # global coupling
    C_12_min, C_12_max      = 0, 0.5      # PFC -> OFC
    C_15_min, C_15_max      = 0, 0.5      # dpThal -> OFC
    #C_16_min, C_16_max      = -0.5, 0.5      # vaThal -> OFC
    
    C_21_min, C_21_max      = 0, 0.5      # OFC -> PFC
    #C_25_min, C_25_max      = -0.5, 0.5      # dpThal -> PFC
    C_26_min, C_26_max      = 0, 0.5      # vaThal -> PFC

    C_31_min, C_31_max      = 0, 0.5      # OFC -> NAcc
    #C_32_min, C_32_max      = -0.5, 0.5      # PFC -> NAcc
    C_34_min, C_34_max      = -0.5, 0.5      # Put -> NAcc
    
    #C_41_min, C_41_max      = -0.5, 0.5      # OFC -> Put
    C_42_min, C_42_max      = 0, 0.5      # PFC -> Put
    C_43_min, C_43_max      = -0.5, 0.5      # Nacc -> Put

    C_53_min, C_53_max      = -0.1, 0.5      # NAcc -> dpThal
    #C_54_min, C_54_max      = -0.5, 0.5      # Put -> dpThal

    #C_63_min, C_63_max      = -0.5, 0.5      # Nacc -> vaThal
    C_64_min, C_64_max      = -0.1, 0.5      # Put -> vaThal
    

    # coupling noises on Ornstein-Uhlenbeck process 
    sigma_C_53_min, sigma_C_53_max  = 0.1, 0.4
    #sigma_C_54_min, sigma_C_54_max  = 0.1, 0.4
    #sigma_C_63_min, sigma_C_63_max  = 0.1, 0.4
    sigma_C_64_min, sigma_C_64_max  = 0.1, 0.4

    eta_C_53_min, eta_C_53_max  = 0, 0.1
    #eta_C_54_min, eta_C_54_max  = 0, 0.1
    #eta_C_63_min, eta_C_63_max  = 0, 0.1
    eta_C_64_min, eta_C_64_max  = 0, 0.1

    prior = pyabc.Distribution(
        sigma = pyabc.RV("uniform", sigma_min, sigma_max - sigma_min),
        G = pyabc.RV("uniform", G_min, G_max - G_min),
        
        C_12 = pyabc.RV("uniform", C_12_min, C_12_max - C_12_min),
        C_15 = pyabc.RV("uniform", C_15_min, C_15_max - C_15_min),
        #C_16 = pyabc.RV("uniform", C_16_min, C_16_max - C_16_min),

        C_21 = pyabc.RV("uniform", C_21_min, C_21_max - C_21_min),
        #C_25 = pyabc.RV("uniform", C_25_min, C_25_max - C_25_min),
        C_26 = pyabc.RV("uniform", C_26_min, C_26_max - C_26_min),

        C_31 = pyabc.RV("uniform", C_31_min, C_31_max - C_31_min),
        #C_32 = pyabc.RV("uniform", C_32_min, C_32_max - C_32_min),
        C_34 = pyabc.RV("uniform", C_34_min, C_34_max - C_34_min),
        
        #C_41 = pyabc.RV("uniform", C_41_min, C_41_max - C_41_min),
        C_42 = pyabc.RV("uniform", C_42_min, C_42_max - C_42_min),
        C_43 = pyabc.RV("uniform", C_43_min, C_43_max - C_43_min),
        
        C_53 = pyabc.RV("uniform", C_53_min, C_53_max - C_53_min),
        #C_54 = pyabc.RV("uniform", C_54_min, C_54_max - C_54_min),
        
        #C_63 = pyabc.RV("uniform", C_63_min, C_63_max - C_63_min),
        C_64 = pyabc.RV("uniform", C_64_min, C_64_max - C_64_min),
        
        sigma_C_53 = pyabc.RV("uniform", sigma_C_53_min, sigma_C_53_max - sigma_C_53_min),
        #sigma_C_54 = pyabc.RV("uniform", sigma_C_54_min, sigma_C_54_max - sigma_C_54_min),
        #sigma_C_63 = pyabc.RV("uniform", sigma_C_63_min, sigma_C_63_max - sigma_C_63_min),
        sigma_C_64 = pyabc.RV("uniform", sigma_C_64_min, sigma_C_64_max - sigma_C_64_min),

        eta_C_53 = pyabc.RV("uniform", eta_C_53_min, eta_C_53_max - eta_C_53_min),
        #eta_C_54 = pyabc.RV("uniform", eta_C_54_min, eta_C_54_max - eta_C_54_min),
        #eta_C_63 = pyabc.RV("uniform", eta_C_63_min, eta_C_63_max - eta_C_63_min),
        eta_C_64 = pyabc.RV("uniform", eta_C_64_min, eta_C_64_max - eta_C_64_min) )

    bounds = dict()
    bounds['sigma'] = [sigma_min, sigma_max]
    bounds['G'] = [G_min, G_max]
    
    bounds['C_12'] = [C_12_min, C_12_max]
    bounds['C_15'] = [C_15_min, C_15_max]
    #bounds['C_16'] = [C_16_min, C_16_max]

    bounds['C_21'] = [C_21_min, C_21_max]
    #bounds['C_25'] = [C_25_min, C_25_max]
    bounds['C_26'] = [C_26_min, C_26_max]

    bounds['C_31'] = [C_31_min, C_31_max]
    #bounds['C_32'] = [C_32_min, C_32_max]
    bounds['C_34'] = [C_34_min, C_34_max]
    
    #bounds['C_41'] = [C_41_min, C_41_max]
    bounds['C_42'] = [C_42_min, C_42_max]
    bounds['C_43'] = [C_43_min, C_43_max]
    
    bounds['C_53'] = [C_53_min, C_53_max]
    #bounds['C_54'] = [C_54_min, C_54_max]
    
    #bounds['C_63'] = [C_63_min, C_63_max]
    bounds['C_64'] = [C_64_min, C_64_max]
    
    bounds['eta_C_53'] = [eta_C_53_min, eta_C_53_max]
    #bounds['eta_C_54'] = [eta_C_54_min, eta_C_54_max]
    bounds['sigma_C_53'] = [sigma_C_53_min, sigma_C_53_max]
    #bounds['sigma_C_54'] = [sigma_C_54_min, sigma_C_54_max]
    #bounds['eta_C_63'] = [eta_C_63_min, eta_C_63_max]
    bounds['eta_C_64'] = [eta_C_64_min, eta_C_64_max]
    #bounds['sigma_C_63'] = [sigma_C_63_min, sigma_C_63_max]
    bounds['sigma_C_64'] = [sigma_C_64_min, sigma_C_64_max]
    return prior, bounds

def get_prior_Thal_hc_weak():
    """ Create uniform prior distributions of parameters to start Sequential Monte-Carlo.
    
    Returns
    -------
        prior: pyABC.Distribution
            Distribution object of priors.
        bounds: dict
            Lower and upper bounds of parameters , i.e. bounds['param_name']=[min,max].
    """

    # PRIORS
    # lower_bound, upper_bound
    sigma_min, sigma_max    = 0.05, 0.1      # noise amplitude
    G_min, G_max            = 2, 3           # global coupling
    C_12_min, C_12_max      = -0.5, 0.5      # PFC -> OFC
    C_15_min, C_15_max      = -0.5, 0.5      # dpThal -> OFC
    C_16_min, C_16_max      = -0.5, 0.5      # vaThal -> OFC
    
    C_21_min, C_21_max      = -0.5, 0.5      # OFC -> PFC
    C_25_min, C_25_max      = -0.5, 0.5      # dpThal -> PFC
    C_26_min, C_26_max      = -0.5, 0.5      # vaThal -> PFC

    C_31_min, C_31_max      = 0, 0.5      # OFC -> NAcc
    #C_32_min, C_32_max      = -0.5, 0.5      # PFC -> NAcc
    C_34_min, C_34_max      = -0.5, 0.5      # Put -> NAcc
    
    #C_41_min, C_41_max      = -0.5, 0.5      # OFC -> Put
    C_42_min, C_42_max      = -0.5, 0.5      # PFC -> Put
    C_43_min, C_43_max      = -0.5, 0.5      # Nacc -> Put

    C_53_min, C_53_max      = -0.5, 0.5      # NAcc -> dpThal
    #C_54_min, C_54_max      = -0.5, 0.5      # Put -> dpThal

    #C_63_min, C_63_max      = -0.5, 0.5      # Nacc -> vaThal
    C_64_min, C_64_max      = -0.5, 0.5      # Put -> vaThal
    

    # coupling noises on Ornstein-Uhlenbeck process 
    sigma_C_53_min, sigma_C_53_max  = 0.1, 0.4
    #sigma_C_54_min, sigma_C_54_max  = 0.1, 0.4
    #sigma_C_63_min, sigma_C_63_max  = 0.1, 0.4
    sigma_C_64_min, sigma_C_64_max  = 0.1, 0.4

    eta_C_53_min, eta_C_53_max  = 0, 0.1
    #eta_C_54_min, eta_C_54_max  = 0, 0.1
    #eta_C_63_min, eta_C_63_max  = 0, 0.1
    eta_C_64_min, eta_C_64_max  = 0, 0.1

    prior = pyabc.Distribution(
        sigma = pyabc.RV("uniform", sigma_min, sigma_max - sigma_min),
        G = pyabc.RV("uniform", G_min, G_max - G_min),
        
        C_12 = pyabc.RV("uniform", C_12_min, C_12_max - C_12_min),
        C_15 = pyabc.RV("uniform", C_15_min, C_15_max - C_15_min),
        C_16 = pyabc.RV("uniform", C_16_min, C_16_max - C_16_min),

        C_21 = pyabc.RV("uniform", C_21_min, C_21_max - C_21_min),
        C_25 = pyabc.RV("uniform", C_25_min, C_25_max - C_25_min),
        C_26 = pyabc.RV("uniform", C_26_min, C_26_max - C_26_min),

        C_31 = pyabc.RV("uniform", C_31_min, C_31_max - C_31_min),
        #C_32 = pyabc.RV("uniform", C_32_min, C_32_max - C_32_min),
        C_34 = pyabc.RV("uniform", C_34_min, C_34_max - C_34_min),
        
        #C_41 = pyabc.RV("uniform", C_41_min, C_41_max - C_41_min),
        C_42 = pyabc.RV("uniform", C_42_min, C_42_max - C_42_min),
        C_43 = pyabc.RV("uniform", C_43_min, C_43_max - C_43_min),
        
        C_53 = pyabc.RV("uniform", C_53_min, C_53_max - C_53_min),
        #C_54 = pyabc.RV("uniform", C_54_min, C_54_max - C_54_min),
        
        #C_63 = pyabc.RV("uniform", C_63_min, C_63_max - C_63_min),
        C_64 = pyabc.RV("uniform", C_64_min, C_64_max - C_64_min),
        
        sigma_C_53 = pyabc.RV("uniform", sigma_C_53_min, sigma_C_53_max - sigma_C_53_min),
        #sigma_C_54 = pyabc.RV("uniform", sigma_C_54_min, sigma_C_54_max - sigma_C_54_min),
        #sigma_C_63 = pyabc.RV("uniform", sigma_C_63_min, sigma_C_63_max - sigma_C_63_min),
        sigma_C_64 = pyabc.RV("uniform", sigma_C_64_min, sigma_C_64_max - sigma_C_64_min),

        eta_C_53 = pyabc.RV("uniform", eta_C_53_min, eta_C_53_max - eta_C_53_min),
        #eta_C_54 = pyabc.RV("uniform", eta_C_54_min, eta_C_54_max - eta_C_54_min),
        #eta_C_63 = pyabc.RV("uniform", eta_C_63_min, eta_C_63_max - eta_C_63_min),
        eta_C_64 = pyabc.RV("uniform", eta_C_64_min, eta_C_64_max - eta_C_64_min) )

    bounds = dict()
    bounds['sigma'] = [sigma_min, sigma_max]
    bounds['G'] = [G_min, G_max]
    
    bounds['C_12'] = [C_12_min, C_12_max]
    bounds['C_15'] = [C_15_min, C_15_max]
    bounds['C_16'] = [C_16_min, C_16_max]

    bounds['C_21'] = [C_21_min, C_21_max]
    bounds['C_25'] = [C_25_min, C_25_max]
    bounds['C_26'] = [C_26_min, C_26_max]

    bounds['C_31'] = [C_31_min, C_31_max]
    #bounds['C_32'] = [C_32_min, C_32_max]
    bounds['C_34'] = [C_34_min, C_34_max]
    
    #bounds['C_41'] = [C_41_min, C_41_max]
    bounds['C_42'] = [C_42_min, C_42_max]
    bounds['C_43'] = [C_43_min, C_43_max]
    
    bounds['C_53'] = [C_53_min, C_53_max]
    #bounds['C_54'] = [C_54_min, C_54_max]
    
    #bounds['C_63'] = [C_63_min, C_63_max]
    bounds['C_64'] = [C_64_min, C_64_max]
    
    bounds['eta_C_53'] = [eta_C_53_min, eta_C_53_max]
    #bounds['eta_C_54'] = [eta_C_54_min, eta_C_54_max]
    bounds['sigma_C_53'] = [sigma_C_53_min, sigma_C_53_max]
    #bounds['sigma_C_54'] = [sigma_C_54_min, sigma_C_54_max]
    #bounds['eta_C_63'] = [eta_C_63_min, eta_C_63_max]
    bounds['eta_C_64'] = [eta_C_64_min, eta_C_64_max]
    #bounds['sigma_C_63'] = [sigma_C_63_min, sigma_C_63_max]
    bounds['sigma_C_64'] = [sigma_C_64_min, sigma_C_64_max]
    return prior, bounds


def get_prior_Thal_fc_weak():
    """ Create uniform prior distributions of parameters to start Sequential Monte-Carlo.
    
    Returns
    -------
        prior: pyABC.Distribution
            Distribution object of priors.
        bounds: dict
            Lower and upper bounds of parameters , i.e. bounds['param_name']=[min,max].
    """

    # PRIORS
    # lower_bound, upper_bound
    sigma_min, sigma_max    = 0.05, 0.1      # noise amplitude
    G_min, G_max            = 2, 3           # global coupling
    C_12_min, C_12_max      = -0.5, 0.5      # PFC -> OFC
    C_15_min, C_15_max      = -0.5, 0.5      # dpThal -> OFC
    C_16_min, C_16_max      = -0.5, 0.5      # vaThal -> OFC
    
    C_21_min, C_21_max      = -0.5, 0.5      # OFC -> PFC
    C_25_min, C_25_max      = -0.5, 0.5      # dpThal -> PFC
    C_26_min, C_26_max      = -0.5, 0.5      # vaThal -> PFC

    C_31_min, C_31_max      = -0.5, 0.5      # OFC -> NAcc
    C_32_min, C_32_max      = -0.5, 0.5      # PFC -> NAcc
    C_34_min, C_34_max      = -0.5, 0.5      # Put -> NAcc
    
    C_41_min, C_41_max      = -0.5, 0.5      # OFC -> Put
    C_42_min, C_42_max      = -0.5, 0.5      # PFC -> Put
    C_43_min, C_43_max      = -0.5, 0.5      # Nacc -> Put

    C_53_min, C_53_max      = -0.5, 0.5      # NAcc -> dpThal
    C_54_min, C_54_max      = -0.5, 0.5      # Put -> dpThal

    C_63_min, C_63_max      = -0.5, 0.5      # Nacc -> vaThal
    C_64_min, C_64_max      = -0.5, 0.5      # Put -> vaThal
    

    # coupling noises on Ornstein-Uhlenbeck process 
    sigma_C_53_min, sigma_C_53_max  = 0.1, 0.4
    sigma_C_54_min, sigma_C_54_max  = 0.1, 0.4
    sigma_C_63_min, sigma_C_63_max  = 0.1, 0.4
    sigma_C_64_min, sigma_C_64_max  = 0.1, 0.4

    eta_C_53_min, eta_C_53_max  = 0, 0.1
    eta_C_54_min, eta_C_54_max  = 0, 0.1
    eta_C_63_min, eta_C_63_max  = 0, 0.1
    eta_C_64_min, eta_C_64_max  = 0, 0.1

    prior = pyabc.Distribution(
        sigma = pyabc.RV("uniform", sigma_min, sigma_max - sigma_min),
        G = pyabc.RV("uniform", G_min, G_max - G_min),
        
        C_12 = pyabc.RV("uniform", C_12_min, C_12_max - C_12_min),
        C_15 = pyabc.RV("uniform", C_15_min, C_15_max - C_15_min),
        C_16 = pyabc.RV("uniform", C_16_min, C_16_max - C_16_min),

        C_21 = pyabc.RV("uniform", C_21_min, C_21_max - C_21_min),
        C_25 = pyabc.RV("uniform", C_25_min, C_25_max - C_25_min),
        C_26 = pyabc.RV("uniform", C_26_min, C_26_max - C_26_min),

        C_31 = pyabc.RV("uniform", C_31_min, C_31_max - C_31_min),
        C_32 = pyabc.RV("uniform", C_32_min, C_32_max - C_32_min),
        C_34 = pyabc.RV("uniform", C_34_min, C_34_max - C_34_min),
        
        C_41 = pyabc.RV("uniform", C_41_min, C_41_max - C_41_min),
        C_42 = pyabc.RV("uniform", C_42_min, C_42_max - C_42_min),
        C_43 = pyabc.RV("uniform", C_43_min, C_43_max - C_43_min),
        
        C_53 = pyabc.RV("uniform", C_53_min, C_53_max - C_53_min),
        C_54 = pyabc.RV("uniform", C_54_min, C_54_max - C_54_min),
        
        C_63 = pyabc.RV("uniform", C_63_min, C_63_max - C_63_min),
        C_64 = pyabc.RV("uniform", C_64_min, C_64_max - C_64_min),
        
        sigma_C_53 = pyabc.RV("uniform", sigma_C_53_min, sigma_C_53_max - sigma_C_53_min),
        sigma_C_54 = pyabc.RV("uniform", sigma_C_54_min, sigma_C_54_max - sigma_C_54_min),
        sigma_C_63 = pyabc.RV("uniform", sigma_C_63_min, sigma_C_63_max - sigma_C_63_min),
        sigma_C_64 = pyabc.RV("uniform", sigma_C_64_min, sigma_C_64_max - sigma_C_64_min),

        eta_C_53 = pyabc.RV("uniform", eta_C_53_min, eta_C_53_max - eta_C_53_min),
        eta_C_54 = pyabc.RV("uniform", eta_C_54_min, eta_C_54_max - eta_C_54_min),
        eta_C_63 = pyabc.RV("uniform", eta_C_63_min, eta_C_63_max - eta_C_63_min),
        eta_C_64 = pyabc.RV("uniform", eta_C_64_min, eta_C_64_max - eta_C_64_min) )

    bounds = dict()
    bounds['sigma'] = [sigma_min, sigma_max]
    bounds['G'] = [G_min, G_max]
    
    bounds['C_12'] = [C_12_min, C_12_max]
    bounds['C_15'] = [C_15_min, C_15_max]
    bounds['C_16'] = [C_16_min, C_16_max]

    bounds['C_21'] = [C_21_min, C_21_max]
    bounds['C_25'] = [C_25_min, C_25_max]
    bounds['C_26'] = [C_26_min, C_26_max]

    bounds['C_31'] = [C_31_min, C_31_max]
    bounds['C_32'] = [C_32_min, C_32_max]
    bounds['C_34'] = [C_34_min, C_34_max]
    
    bounds['C_41'] = [C_41_min, C_41_max]
    bounds['C_42'] = [C_42_min, C_42_max]
    bounds['C_43'] = [C_43_min, C_43_max]
    
    bounds['C_53'] = [C_53_min, C_53_max]
    bounds['C_54'] = [C_54_min, C_54_max]
    
    bounds['C_63'] = [C_63_min, C_63_max]
    bounds['C_64'] = [C_64_min, C_64_max]
    
    bounds['eta_C_53'] = [eta_C_53_min, eta_C_53_max]
    bounds['eta_C_54'] = [eta_C_54_min, eta_C_54_max]
    bounds['sigma_C_53'] = [sigma_C_53_min, sigma_C_53_max]
    bounds['sigma_C_54'] = [sigma_C_54_min, sigma_C_54_max]
    bounds['eta_C_63'] = [eta_C_63_min, eta_C_63_max]
    bounds['eta_C_64'] = [eta_C_64_min, eta_C_64_max]
    bounds['sigma_C_63'] = [sigma_C_63_min, sigma_C_63_max]
    bounds['sigma_C_64'] = [sigma_C_64_min, sigma_C_64_max]
    return prior, bounds


def get_prior_scan_con():
    """ Create uniform prior distributions of parameters to start Sequential Monte-Carlo.
    
    Returns
    -------
        prior: pyABC.Distribution
            Distribution object of priors.
        bounds: dict
            Lower and upper bounds of parameters , i.e. bounds['param_name']=[min,max].
    """

    # PRIORS
    # lower_bound, upper_bound
    sigma_min, sigma_max    = 0.05, 0.1     # noise amplitude
    G_min, G_max            = 2, 3           # global coupling
    C_12_min, C_12_max      = -0.5, 0.5        # CON -> SCAN
    C_21_min, C_21_max      = -0.5, 0.5        # SCAN -> CON
    C_13_min, C_13_max      = -0.1, 0.5      # NAcc -> SCAN
    C_31_min, C_31_max      = 0, 0.5         # SCAN -> NAcc
    C_32_min, C_32_max      = 0, 0.5         # CON -> NAcc
    C_23_min, C_23_max      = -0.1, 0.5         # NAcc -> CON
    C_24_min, C_24_max      = -0.1, 0.5      # Put -> CON
    C_42_min, C_42_max      = 0, 0.5         # CON -> Put
    C_41_min, C_41_max      = 0, 0.5      # SCAN -> Put
    C_14_min, C_14_max      = -0.1, 0.5      # Put -> SCAN
    C_34_min, C_34_max      = -0.5, 0.5      # Put -> NAcc
    C_43_min, C_43_max      = -0.5, 0.5      # Nacc -> Put

    # coupling noises on Ornstein-Uhlenbeck process 
    sigma_C_13_min, sigma_C_13_max  = 0.1, 0.4
    sigma_C_23_min, sigma_C_23_max  = 0.1, 0.4
    sigma_C_24_min, sigma_C_24_max  = 0.1, 0.4
    sigma_C_14_min, sigma_C_14_max  = 0.1, 0.4
    eta_C_13_min, eta_C_13_max  = 0, 0.1
    eta_C_23_min, eta_C_23_max  = 0, 0.1
    eta_C_24_min, eta_C_24_max  = 0, 0.1
    eta_C_14_min, eta_C_14_max  = 0, 0.1


    prior = pyabc.Distribution(
        sigma = pyabc.RV("uniform", sigma_min, sigma_max - sigma_min),
        G = pyabc.RV("uniform", G_min, G_max - G_min),
        C_12 = pyabc.RV("uniform", C_12_min, C_12_max - C_12_min),
        C_21 = pyabc.RV("uniform", C_21_min, C_21_max - C_21_min),
        C_13 = pyabc.RV("uniform", C_13_min, C_13_max - C_13_min),
        C_23 = pyabc.RV("uniform", C_23_min, C_23_max - C_23_min),
        C_32 = pyabc.RV("uniform", C_32_min, C_32_max - C_32_min),
        C_31 = pyabc.RV("uniform", C_31_min, C_31_max - C_31_min),
        C_41 = pyabc.RV("uniform", C_41_min, C_41_max - C_41_min),
        C_14 = pyabc.RV("uniform", C_14_min, C_14_max - C_14_min),
        C_24 = pyabc.RV("uniform", C_24_min, C_24_max - C_24_min),
        C_42 = pyabc.RV("uniform", C_42_min, C_42_max - C_42_min),
        C_34 = pyabc.RV("uniform", C_34_min, C_34_max - C_34_min),
        C_43 = pyabc.RV("uniform", C_43_min, C_43_max - C_43_min),
        sigma_C_13 = pyabc.RV("uniform", sigma_C_13_min, sigma_C_13_max - sigma_C_13_min),
        sigma_C_23 = pyabc.RV("uniform", sigma_C_23_min, sigma_C_23_max - sigma_C_23_min),
        sigma_C_24 = pyabc.RV("uniform", sigma_C_24_min, sigma_C_24_max - sigma_C_24_min),
        sigma_C_14 = pyabc.RV("uniform", sigma_C_14_min, sigma_C_14_max - sigma_C_14_min),
        eta_C_13 = pyabc.RV("uniform", eta_C_13_min, eta_C_13_max - eta_C_13_min),
        eta_C_23 = pyabc.RV("uniform", eta_C_23_min, eta_C_23_max - eta_C_23_min),
        eta_C_14 = pyabc.RV("uniform", eta_C_14_min, eta_C_14_max - eta_C_14_min),
        eta_C_24 = pyabc.RV("uniform", eta_C_24_min, eta_C_24_max - eta_C_24_min) )

    bounds = dict()
    bounds['sigma'] = [sigma_min, sigma_max]
    bounds['G'] = [G_min, G_max]
    bounds['C_12'] = [C_12_min, C_12_max]
    bounds['C_21'] = [C_21_min, C_21_max]
    bounds['C_13'] = [C_13_min, C_13_max]
    bounds['C_23'] = [C_23_min, C_23_max]
    bounds['C_31'] = [C_31_min, C_31_max]
    bounds['C_32'] = [C_32_min, C_32_max]
    bounds['C_41'] = [C_41_min, C_41_max]
    bounds['C_24'] = [C_24_min, C_24_max]
    bounds['C_42'] = [C_42_min, C_42_max]
    bounds['C_14'] = [C_14_min, C_14_max]
    bounds['C_34'] = [C_34_min, C_34_max]
    bounds['C_43'] = [C_43_min, C_43_max]
    bounds['eta_C_13'] = [eta_C_13_min, eta_C_13_max]
    bounds['eta_C_23'] = [eta_C_23_min, eta_C_23_max]
    bounds['sigma_C_13'] = [sigma_C_13_min, sigma_C_13_max]
    bounds['sigma_C_23'] = [sigma_C_23_min, sigma_C_23_max]
    bounds['eta_C_24'] = [eta_C_24_min, eta_C_24_max]
    bounds['eta_C_14'] = [eta_C_14_min, eta_C_14_max]
    bounds['sigma_C_24'] = [sigma_C_24_min, sigma_C_24_max]
    bounds['sigma_C_14'] = [sigma_C_14_min, sigma_C_14_max]
    return prior, bounds


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
    model_params, sim_params, control_params, bold_params = get_default_params()

    # unpack model's params
    for k,v in in_params.items():
        if k in model_params.keys():
            if type(v) is np.ndarray:
                v = v.squeeze() # otherwise matrix multiplication can be a problem at inference
            model_params[k] = v
        elif ('C_' in k):
            var, inds = k.rsplit('_', maxsplit=1)
            j,i = int(inds[0])-1, int(inds[1])-1
            model_params[var][j,i] = v

        # unpack sim params
        elif k in sim_params.keys():
            sim_params[k] = v
        
        # unpack bold params
        elif k in bold_params.keys():
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
    """ Run a pool of simulations and score their outputs.

    As a design choice, the number of simulations per pool and the number of processes used in parallel are hard coded here 
    in a `Argparse.Namespace` object which is propagated to the launcher.  
    
    Parameters
    ----------
        params: dict
            Set of parameters are used to instanciate model.

    Returns
    -------
        RMSE: dict
            Root Mean Square Error between the simulated pool (i.e. a "population cohort") 
            and the real data (either "controls" or "patients").
    """
    args = argparse.Namespace()
    args.n_sims = 56
    args.n_jobs = 8
    args.model_pars, args.sim_pars, args.control_pars, args.bold_pars = unpack_params(params)
    
    #sim_objs = parallel_launcher.launch_simulations(args)
    sim_objs = parallel_launcher.launch_pool_simulations(args)
    
    #RMSE = RWW.score_population_models(sim_objs, cohort='controls')
    RMSE = RWW.score_population_models(sim_objs, cohort='patients')
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


def get_config(args):
    """ extract redis config """
    cfg = dict()
    with open(os.path.join(proj_dir, 'envs', 'redis.conf'), 'r') as f:
        line = f.readline()
        cfg['pssr'] = line.split(' ')[1].strip()
    if args.redis_ip==None:
        with open(os.path.join(proj_dir, 'traces', '.redis_ip'), 'r') as f:
            cfg['redis_ip'] = f.readline().strip()
    else:
        cfg['redis_ip'] = args.redis_ip
    return cfg


def parse_args():
    """ Parse command line arguments """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--redis_ip', type=str, default=None, action='store', help='set ip address of the server. default: hpcapp01')
    parser.add_argument('--redis_pwd', type=str, default='bayesopt1234321', action='store', help='password to access server')
    parser.add_argument('--resume_opt', type=str, default=None, action='store', help='name of the optimzation to resume (default:None, start a new optimization)')
    args = parser.parse_args()
    return args

def run_abc(prior, cfg):
    """ Setup and run the Approximate Bayesian Computation.

    Parameters
    ----------
        prior: pyABC.Distribution
            Prior distributions of parameters.
        cfg: Argparse.Namespace
            Server configuration information.

    Returns
    -------
        history: pyABC.History
            Output of the optimization, i.e. population parameters of accepted particles and summary statistics. 
    """

    trace_name = 'rww4D_OU_HPC_SCAN_CON'+today()
    #sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=1)  # used for calibrating on local machine before portage to HPC
    sampler = pyabc.sampler.RedisEvalParallelSampler(host=cfg['redis_ip'], port=6379, password=cfg['pssr'])
    sampler.daemon = False
    abc = pyabc.ABCSMC(simulate_population_rww, prior, RWW.distance, sampler=sampler, population_size=100, max_nr_recorded_particles=5000)
    if args.resume_opt==None:
        abc_id = abc.new(
            "sqlite:///" + os.path.join(proj_dir, 'traces', trace_name+".db"),
            {"err": 0}  # observation # note: here is dummy, the distance function does not use it.
        )
    else:
        abc.load("sqlite:///" + os.path.join(proj_dir, 'traces', args.resume_opt+".db"), abc_id=1)
    history = abc.run(max_nr_populations=10, minimum_epsilon=0.01)
    return history


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    #prior, _ = get_prior()
    #prior, _ = get_prior_Thal()  # <-- hc_strong
    #prior, _ = get_prior_Thal_hc_weak()
    #prior, _ = get_prior_Thal_fc_weak()
    prior, _ = get_prior_scan_con()
    history = run_abc(prior, cfg)