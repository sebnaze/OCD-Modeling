import numpy as np

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
