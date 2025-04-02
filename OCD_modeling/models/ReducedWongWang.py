import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import pandas as pd
import pickle
import platform
import scipy
import scipy.stats
import sklearn
from sklearn import preprocessing
import time

from OCD_modeling.utils.utils import get_working_dir
#from OCD_modeling.models.HemodynamicResponseModeling.BalloonWindkessel import balloonWindkessel
#from OCD_modeling.utils.neurolib.neurolib.models.bold.timeIntegration import simulateBOLD
#from OCD_modeling.utils import simulateBOLD
from neurolib.models.bold import simulateBOLD

# get computer name to set paths
working_dir = get_working_dir()

# general paths
proj_dir = working_dir+'lab_lucac/sebastiN/projects/OCD_modeling'

class ReducedWongWang:
    """ Reduced Wong Wang model (1-dimensional) """
    def __init__(self, a=270., b=108., d=0.154,
                 I_0=0.3, J_N=0.2609, w=0.9, G=1., C=0., S_j=0.,
                 tau_S=100., gamma=0.000641, sigma=0.001, v_i=0.):

        # synaptic gating params
        self.a = a             # slope (n/C); default=270
        self.b = b             # offset (Hz); default=108
        self.d = d             # decay (s);   default=0.154

        # firing rate params
        self.I_0 = I_0         # external input (nA); default=0.3
        self.J_N = J_N         # synaptic coupling (nA); default=0.2609
        self.w = w             # local exc recurrence (n/a); default=0.9
        self.G = G             # global scaling factor (n/a); default=1
        self.C = C             # connectivity matrix (n/a); default=0
        self.S_j = S_j         # coupled population firing rates (Hz); default=0

        # ODE params
        self.tau_S = tau_S      # kinetic parameter of local population (ms); default=100
        self.gamma = gamma     # kinetic parameter of coupled population (ms); default=0.000641
        self.sigma = sigma     # noise amplitude (in node) (nA); default=0.001
        self.v_i = v_i         # gaussian noise (n/a); default=0

    def H(self, x):
        """ Average synaptic gating.
    
        a: slope (n/C); default=270
        b: offset (Hz); default=108
        d: decay (s);   default=0.154
        
        """
        return (self.a * x - self.b) / (1. - np.exp(-self.d * (self.a * x - self.b)))

    def S_i(self, x, S_j=0):
        """ Firing rate.
    
        I_0: external input (nA); default=0.3
        J_N: synaptic coupling (nA); default=0.2609
        w: local exc recurrence (n/a); default=0.9
        G: global scaling factor (n/a); default=1
        C: connectivity matrix (n/a); default=0
        S_j: coupled population firing rates (Hz); default=0
    
            """
        return (x - self.I_0 - self.G * self.J_N * self.C * S_j) / (self.w * self.J_N)

    def dS_i(self, x, v_i=0):
        """ ODE of firing rate.
        
        tau_S: kinetic parameter of local population (ms); default=100
        gamma: kinetic parameter of coupled population (ms); default=0.000641
        sigma: noise amplitude (nA); default=0.001
        v_i: gaussian noise (n/a); default=0
        
        """
        return (-S_i(x)/self.tau_S + (1 - S_i(x)) * self.gamma * H(x) + self.sigma*v_i)

    def S_nc(self, x):
        " Nullcline (dS/dt=0) "
        return (-self.tau_S*self.gamma*H(x) / (-1.-self.tau_S*self.gamma*H(x)))





class ReducedWongWangND:
    """ Reduced Wong Wang model (N-dimensional) """
    def __init__(self, a=270., b=108., d=0.154,
                 I_0=0.3, J_N=0.2609, w=0.9, G=1.,
                 tau_S=100., gamma=0.000641, sigma=0.001, N=2, dt=0.01,
                 C=None, S=None, x=None, *args, **kwargs):

        # synaptic gating params
        self.a = a             # slope (n/C); default=270
        self.b = b             # offset (Hz); default=108
        self.d = d             # decay (s);   default=0.154

        # firing rate params
        if type(I_0)==np.ndarray:
            self.I_0 = I_0
        else:
            self.I_0 = np.ones(N,)*I_0
        #self.I_0 = I_0         # external input (nA); default=0.3
        self.J_N = J_N         # synaptic coupling (nA); default=0.2609
        self.w = w             # local exc recurrence (n/a); default=0.9
        self.G = G             # global scaling factor (n/a); default=1

        if S is None:
            self.S = np.array(np.random.rand(N,)*0.1+0.05, dtype=np.float64)         # coupled population firing rates (Hz);
        else:
            self.S = np.array(S).squeeze().astype(np.float64)
        self.x = np.array(np.random.rand(N,)*0.1+0.05, dtype=np.float64)         # coupled population activity (Hz); default = uniformly distributed

        # ODE params
        self.tau_S = tau_S      # kinetic parameter of local population (ms); default=100
        self.gamma = gamma     # kinetic parameter of coupled population (ms); default=0.000641
        self.sigma = sigma     # noise amplitude (nA); default=0.001
        #self.v = v_0         # gaussian noise (n/a); default=0
        self.dt = dt
        self.N = N          # number of units

        if C is None:
            self.C = np.random.randn(N,N) * (1-np.eye(N,N))  # connectivity matrix (n/a); default= 0 self, 1 to others
        elif type(C)==list:
            self.C = np.array(C)
        else:
            self.C = C
        
        self.control_params = {}  # initialize empty control parameters, they will be set after the model is instanciated

    def H(self, x):
        """ Average synaptic gating.

        a: slope (n/C); default=270
        b: offset (Hz); default=108
        d: decay (s);   default=0.154

        """

        return (self.a * x - self.b) / (1. - np.exp(-self.d * (self.a * x - self.b)))

    def v(self):
        """ Implementing the Gaussian white noise process. """
        return np.array(np.random.randn(self.N,), dtype=np.float64)

    def dS(self):
        """ ODE of firing rate.
        
        tau_S: kinetic parameter of local population (ms); default=100
        gamma: kinetic parameter of coupled population (ms); default=0.000641
        sigma: noise amplitude (nA); default=0.001
        v_i: gaussian noise (n/a); default=0

        """
        return (-self.S/self.tau_S + (1 - self.S) @ (self.gamma * self.H(x=self.x)) + self.sigma*self.v())

    
    def integrate(self):
        """ Euler(-Maruyama) integration of the ODE """
        self.x = self.w * self.J_N * self.S + self.G * self.J_N * self.C @ self.S + self.I_0
        H = self.H(x=self.x)
        v = self.v()
        dS = (-self.S/self.tau_S + (1 - self.S) * (self.gamma * H) + self.sigma*v)
        self.S = self.S + dS*self.dt

    def run(self, t_tot=1000, sf=100, t_rec=None, rec_vars=[]):
        """ Runs the model.

        Parameters
        ----------
            t_tot: int
                Total simulation time (s).
            sf: int
                Sampling frequency of the reccording (Hz).
            t_rec: list
                Interval of recording (s) in the form ``[start, stop]``.
            rec_vars: list
                Variables to records (note that S is always recorded).

        """
        n_ts = int(t_tot/self.dt)
        sf_dt = 1./(sf*self.dt)

        # fix sf if needed based on dt
        if sf_dt.is_integer():
            self.sf = sf
        else:
            # case of sf higher than dt permits
            if sf_dt<1:
                sf_dt=1
            # case of sf not multiple of dt
            else:
                sf_dt = np.floor(sf_dt)
            self.sf = 1. / self.dt / sf_dt
            print("Sampling frequency not a multiple of dt, used sf={} instead".format(self.sf))

        # set recording time if unspecified
        if t_rec==None:
            t_rec = [0,t_tot]
        n_rec = int((t_rec[1]-t_rec[0])*self.sf)  # number of steps recorded
        
        # prepare variables to record
        self.t = np.arange(t_rec[0],t_rec[1], 1./self.sf)
        self.S_rec = np.zeros((n_rec,self.N))

        self.prepare_auxiliary_variables(rec_vars, n_rec)
        self.prepare_control_params()

        # run the model
        rec_idx = 0
        for i in range(n_ts):
            self.integrate()
            # record and update control parameters
            if ((not i%sf_dt) & ((i*self.dt)>=t_rec[0]) & ((i*self.dt)<=t_rec[1])):
                self.S_rec[rec_idx,:] = self.S.copy()
                self.record_auxiliary_variables(rec_vars, rec_idx)
                self.update_control_params(rec_idx)
                rec_idx += 1


    def set_control_params(self, params:dict):
        """ Set the parameters to be updated during the simulation (e.g. a slow control parameter).
        
        Parameters
        ----------
            params: dict
                Parameters to be udpated, keys of this dict must match parameters names of the model.
                values of the dictionary are list of tuple indicating times and values of the parameter to be updated, i.e.:
                ``params = {I_0: [ (t0,v0), (t1,v1), (t2,v2), ... ]}``

        Note that the update is linear monotonic between referenced points, and the update frequency used is 
        the sampling frequency (SF). Control parameter can only be changed during recording period.

        """
        for k,v in params.items():
            if not hasattr(self,k):
                if not k.startswith('C_'):
                    print(f"Control parameter {k} does not exist in model, it won't be updated")
                    params.pop(k)
        self.control_params = params
                

    def load_C_param(self, par, vals):
        """ Special case for connectivity control parameters as it is an array """
        # load control param if already exists (i.e. other C indices have been set), otherwise create it
        if hasattr(self, 'update_C'):
            update_par = self.update_C
        else:
            update_par = np.array([self.C.copy()  for _ in range(self.t.shape[0])]) #initialize control param to default value
        ij = par.split('_')[1]
        i,j = int(ij[0])-1,int(ij[1])-1
        ind_0 = 0
        val_0 = update_par[0][i,j]
        for t,v in vals:
            if ( (self.t.min()>t) or (self.t.max()<(t-1)) ):
                print(f"Control parameter {par} cannot be changed if simulation is not recorded during those set times, change recording times or parameter update time ")
                break
            ind_t = np.abs(self.t - t).argmin()  # get closest t index
            n = ind_t - ind_0 # number of values to fill between times
            new_vals = np.linspace(val_0,v,n)
            for k,C in enumerate(update_par[ind_0:ind_t]):
                C[i,j] = new_vals[k] 
            ind_0,val_0 = ind_t,v
        self.update_C = update_par


    def prepare_control_params(self):
        """ Prepare the control parameters to be updated during the simulation """
        for par,vals in self.control_params.items():
            vals.sort() # makes sure the times are set in increasing ordered 
            if par.startswith('C_'):
                self.load_C_param(par,vals)
            else:
                update_par = np.ones(self.t.shape) * getattr(self, par)  #initialize control param to default value
                ind_0 = 0
                val_0 = update_par[0]
                for t,v in vals:
                    if ( (self.t.min()>t) or (self.t.max()<t) ):
                        print(f"Control parameter {par} cannot be changed if simulation is not recorded during those set times, change recording times or parameter update time ")
                        break
                    ind_t = np.abs(self.t - t).argmin()  # get closest t index
                    n = ind_t - ind_0 # number of values to fill between times
                    update_par[ind_0:ind_t] = np.linspace(val_0,v,n)
                    ind_0,val_0 = ind_t,v
                setattr(self, 'update_'+par, update_par)


    def prepare_auxiliary_variables(self, rec_vars, n_rec):
        """ creates dtata structures to recorded supplementary variables """
        for var in rec_vars:
            if hasattr(self, var):
                setattr(self, 'rec_'+var, np.zeros((n_rec,))) # <-- assumes single variable, if vector or matrice it needs different shape
            elif var.startswith('C_'):
                setattr(self, 'rec_'+var, np.zeros((n_rec,)))
            else:
                print(f"Variable {var} does not seem to exist, it cannot be recorded.")
                rec_vars.remove(var)


    def record_auxiliary_variables(self, rec_vars, rec_idx):
        """ Record variables other than S """
        for var in rec_vars:
            if hasattr(self, var):
                val = getattr(self, var)                  
            elif var.startswith('C_'):
                ij = var.split('_')[1]
                i,j = int(ij[0])-1,int(ij[1])-1
                val = self.C[i,j]
            else:
                continue
            rec_var = getattr(self, 'rec_'+var)
            rec_var[rec_idx] = val.copy()


    def update_control_params(self, index):
        """ Update the control parameters during the simulation """
        for par in self.control_params.keys():
            # drawback of connectivity control parameters:  C is loaded n times if n C_ij's are modidfied at the same time (not a big deal) 
            if par.startswith('C_'):
                par = 'C'
            update_par = getattr(self, 'update_'+par)
            setattr(self, par, update_par[index])


class ReducedWongWangOU(ReducedWongWangND):
    """ Reduced Wong-Wang model with Ornsetin-Uhlenbeck process for coupling (n dimensions) """
    def __init__(self, N=4, sigma_C=[], eta_C=[], *args, **kwargs):
        super().__init__(N=N, *args, **kwargs)
        self.vC = self.C.copy()             # variable connectivity variables
        if len(sigma_C)==0:
            self.sigma_C = np.zeros((N,N))
        else:
            self.sigma_C = np.array(sigma_C)
        if len(eta_C)==0:
            self.eta_C = np.zeros((N,N))
        else:
            self.eta_C = np.array(eta_C)

    def integrate(self):
        """ Euler(-Maruyama) integration of the ODE """
        self.x = self.w * self.J_N * self.S + self.G * self.J_N * self.vC @ self.S + self.I_0
        H = self.H(x=self.x)
        v = self.v()
        dS = (-self.S/self.tau_S + (1 - self.S) * (self.gamma * H) + self.sigma*v)
        dvC = -self.eta_C*(self.vC - self.C) + self.sigma_C*np.random.randn(self.N, self.N)
        self.S = self.S + dS*self.dt
        self.vC = self.vC + dvC*self.dt


    def record_auxiliary_variables(self, rec_vars, rec_idx):
        """ Record auxiliary variables other than S """
        for var in rec_vars:
            if hasattr(self, var):
                val = getattr(self, var)                  
            elif var.startswith('C_'):
                ij = var.split('_')[1]
                i,j = int(ij[0])-1,int(ij[1])-1
                val = self.vC[i,j]
            else:
                continue
            rec_var = getattr(self, 'rec_'+var)
            rec_var[rec_idx] = val.copy()

                

#  POST PROCESSING FUNCTIONS  #
# --------------------------- #
def compute_bold(model, t_range=None, transient=30):
    """ BOLD timeseries and functional connectivity between regions.

    Parameters
    ----------
        model: OCD_modeling.models.ReducedWongWang.
            Model object.
        t_range: list
            Times of interest (in sec), in the form ``[start, stop]``. Default: all recorded time.
        transient: int
            Time discarded at the beginning of t_range due to BOLD transient (in sec). Default: 30s.

    """
    inds = get_inds(model, t_range)
    #bold_ts, s, f, v, q = balloonWindkessel(model.S_rec[inds,:].T, 1./model.sf)
    #scaler = sklearn.preprocessing.MinMaxScaler()
    #ts = scaler.fit_transform(model.S_rec[inds,:])
    ts = model.S_rec[inds,:]
    bold_ts, x, f, q, v = simulateBOLD(ts.T, 1./model.sf, voxelCounts=None)
    model.bold_ts = bold_ts[:,int(model.sf*transient):] # discard first 10 sec due to transient
    model.bold_t = model.t[inds[int(model.sf*transient):]]
    model.bold_fc = np.corrcoef(model.bold_ts)


def compute_transitions(model, threshold=0.3, min_diff=3, t_range=None):
    """ Compute the number of transitions between low and high activity states """
    inds = get_inds(model, t_range)
    ts = model.S_rec[inds,:]
    ts_thr = ts > threshold
        
    transitions = dict()
    
    for i in range(model.N):
        bin_ts = ts_thr[:,i]    
        trans = np.where((np.roll(bin_ts,1) != bin_ts))[0] / model.sf
        # handle supra-threshold initial conditions, need to remove first element
        if len(trans)>1:
            if trans[0]==0:
                trans = trans[1:]
            trans_inds = np.where(np.diff(trans)>min_diff)[0]+1
            trans_inds = np.concatenate([[0], trans_inds])
        elif len(trans)==1:
            if trans[0]==0:
                trans = []
                trans_inds = []
            else:
                trans_inds = [0]
        elif len(trans)==0:
            trans_inds = []
            
        transitions['S'+str(i+1)] = np.array(trans[trans_inds], dtype=int)
    
    model.transitions = transitions


def compute_strFr_stats(model, t_range=None, thr=0, rec_vars=['C_12']):
    """ Compute transition times, dwell times, and number of transitions in striato-frontal circuit(s) """
    inds = get_inds(model, t_range)
    output = dict()
    for var in rec_vars:
        ts = getattr(model, 'rec_'+var)
        ts_thr = ts > thr
        transitions = np.diff(ts_thr) # 1 = indirect to direct; -1: direct to indirect
        transitions_inds, = np.where(transitions!=0)
        transitions_times = model.t[transitions_inds]
        dwell_time = np.diff(transitions_times)
        dwell_times = dict()
        if ts_thr[0]>0: # starts direct
            dwell_times['direct'] = dwell_time[0::2]
            dwell_times['indirect'] = dwell_time[1::2]
        else: # starts indirect
            dwell_times['indirect'] = dwell_time[0::2]
            dwell_times['direct'] = dwell_time[1::2]
        output[var] = {'n_transitions':len(transitions_inds), 'transitions_times':transitions_times, 'dwell_times':dwell_times}
    model.strFr_stats = output
        

def create_sim_df(sim_objs, sim_type = 'sim-con', offset=0, dataset='OCD_baseline'):
    """ Make a pandas DataFrame from list of simulation outputs objects """
    if dataset=='OCD_baseline':
        if sim_objs[0].N == 4:
            var_names = ['OFC', 'PFC', 'NAcc', 'Put']
        if sim_objs[0].N == 4:
            var_names = ['OFC', 'PFC', 'NAcc', 'Put']
            pathway_map = {'OFC-PFC': 'OFC_PFC', 'OFC-NAcc': 'Acc_OFC', 'OFC-Put':'dPut_OFC', 'PFC-NAcc':'Acc_PFC', 'PFC-Put':'dPut_PFC', 'NAcc-Put':'Acc_dPut'}
        
        elif sim_objs[0].N == 6:
            var_names = ['OFC', 'PFC', 'NAcc', 'Put', 'DP', 'VA']
            pathway_map = {'OFC-PFC': 'OFC_PFC', 'OFC-NAcc': 'Acc_OFC', 'OFC-Put':'dPut_OFC', 'OFC-DP':'dpThal_OFC', 'OFC-VA':'vaThal_OFC',
                        'PFC-NAcc':'Acc_PFC', 'PFC-Put':'dPut_PFC', 'PFC-DP':'dpThal_PFC', 'PFC-VA':'vaThal_PFC',
                        'NAcc-Put':'Acc_dPut', 'NAcc-DP':'Acc_dpThal', 'NAcc-VA':'Acc_vaThal',
                        'Put-DP':'dPut_dpThal', 'Put-VA':'dPut_vaThal',
                        'DP-VA': 'vaThal_dpThal'}
        else:
            print('Cannot create sim_df if N!=4 or N!=6')
            return
    elif dataset=='OCD_SCAN_CON':
        if sim_objs[0].N == 4:
            var_names = ['SCAN', 'CON', 'Acc', 'Put']
            pathway_map = {'SCAN-CON': 'SCAN_CON', 'SCAN-Acc': 'SCAN_Acc', 'SCAN-Put':'SCAN_Put', 'CON-Acc':'CON_Acc', 'CON-Put':'CON_Put', 'Acc-Put':'Acc_Put'}
        if sim_objs[0].N == 2:
            var_names = ['SCAN', 'CON']
            pathway_map = {'SCAN-CON': 'SCAN_CON'}
        
    lines = []
    for i,sim in enumerate(sim_objs):
        fc = sim.bold_fc
        for j in np.arange(sim.N):
            for k in np.arange(j+1,sim.N):
                val = fc[j,k]
                c = '-'.join([var_names[j], var_names[k]])
                pathway = pathway_map[c]
                line = dict()
                line['subj'] =  sim_type+'{:06d}'.format(offset+i+1)
                line['cohort'] = sim_type
                line['pathway'] = pathway
                line['corr'] = val
                lines.append(line)

    df_sim_fc = pd.DataFrame(lines)
    return df_sim_fc
    


def distance(x,y):
    """ distance to minimize based on score """
    #return 1 - x['r'] + x['corr_diff']
    return x['err']


def get_inds(model, t_range=None):
    """ extract time series indices of interest beased on t_range (in sec) """
    if t_range==None:
        t_range=[model.t.min(), model.t.max()]
    inds, = np.where((model.t>=t_range[0]) & (model.t<=t_range[1]))
    return inds


def score_model(rww, coh='con'):
    """ Score single model against empirical FC (only considering mean).
    
    :param rww: instance of model to score.
    :param coh: cohort to be scored against ('con' or 'pat').
    (that is used when optimizing single models, not populations of models)
    
    """
    # load empirical FC
    with open(os.path.join(proj_dir, 'postprocessing', 'R.pkl'), 'rb') as f:
        R = pickle.load(f)

    # compute score
    output = dict()
    fix_inds = [2,3,0,1] # NAcc Put OFC PFC -> OFC PFC NAcc Put
    fixed_inds = np.ix_(fix_inds,fix_inds)
    triu_inds = np.triu_indices(rww.N,k=1)
    corr = rww.bold_fc
    corrData, corrModel = R[coh][fixed_inds][triu_inds].flatten(), corr[triu_inds].flatten()
    corr_MAE = np.sum(np.abs(corrData - corrModel))/len(corrData)
    corr_RMSE = np.sqrt(np.sum((corrData - corrModel)**2)/len(corrData))
    r,pval = scipy.stats.pearsonr(corrData, corrModel)
    return r, corr_MAE, corr_RMSE

def score_population_models(sim_objs, cohort='controls', dataset='OCD_baseline'):
    """ Score a population of simulated model (using a parameter set) against experimental observations.
    Here, the whole distribution of models outputs is scored against the distributions of observations. """
    # load empirical FC
    if dataset=='OCD_baseline':
        if sim_objs[0].N==4:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_avg_2023.pkl'), 'rb') as f:
                df_roi_corr = pickle.load(f)
        if sim_objs[0].N==6:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_roi_corr_avg_2024_Thal.pkl'), 'rb') as f:
                df_roi_corr = pickle.load(f)
    elif dataset=='OCD_SCAN_CON':
        if sim_objs[0].N==4:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_scan_con_2025.pkl'), 'rb') as f:
                df_roi_corr = pickle.load(f)
        if sim_objs[0].N==2:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_scan_con_2025.pkl'), 'rb') as f:
                df_roi_corr = pickle.load(f)
            df_roi_corr = df_roi_corr[df_roi_corr.pathway=='SCAN_CON']
    else:
        return NameError('Dataset {} is unknown.')


    # create simulated FC dataframe
    sim_type = 'sim-'+cohort
    df_sim_fc = create_sim_df(sim_objs, sim_type=sim_type, dataset=dataset)
    df = df_roi_corr[df_roi_corr.cohort==cohort].merge(df_sim_fc, how='outer')
    
    # compute root mean square error
    err = [] 
    for pathway in df.pathway.unique():
        obs = df[(df.pathway==pathway) & (df.cohort==cohort)]['corr']
        sim = df[(df.pathway==pathway) & (df.cohort==sim_type)]['corr']
        err.append((np.mean(obs)-np.mean(sim))**2)
        err.append((np.std(obs)-np.std(sim))**2)
    err = np.sqrt(np.sum(err)/len(err))
    return err


#  PLOTTING FUNCTIONS  #
#----------------------#
def plot_timeseries(model, t_range=None, labels=['OFC', 'PFC', 'NAcc', 'Put']):
    """ visualize time serie generated by model
    
    :param model: ReducedWangWang object.
    
    """
    plt.figure(figsize=[16,4])
    inds = get_inds(model, t_range)
    plt.plot(model.t[inds],model.S_rec[inds,:])
    #plt.legend([str(i) for i in range(model.N)])
    plt.legend(labels, loc='upper left', bbox_to_anchor=[1,1])
    #plt.title("sigma={:.4f} G={:.1f} C={}".format(model.sigma, model.G, model.C))
    plt.show()

def plot_control_params(model, t_range=None, labels=[]):
    """ Visualize time serie of control parameters.
                
    :param model: ReducedWangWang object.

    """
    n_pars = len(list(model.control_params.keys()))
    inds = get_inds(model, t_range)
    plt.figure(figsize=[16,2*n_pars])
    for k,(par,vals) in enumerate(model.control_params.items()):
        if par.startswith('C_'):
            ij = par.split('_')[1]
            i,j = int(ij[0])-1,int(ij[1])-1
            ts = model.update_C[inds][:,i,j]
        else:
            update_par = getattr(model, 'update_'+par)
            ts = update_par[inds]
        plt.subplot(n_pars,1, k+1)
        plt.plot(model.t[inds], ts)
        #plt.legend([str(i) for i in range(model.N)])
        plt.legend([par], loc='upper right')
    plt.show()

def plot_auxiliary_variables(model, t_range=None, rec_vars=[]):
    """ Visualize time serie generated by model
                
    :param rww: ReducedWangWang object.
                
    """
    n = len(rec_vars)
    if n>0:
        plt.figure(figsize=[16,2*n])
        inds = get_inds(model, t_range)
        for var in rec_vars:
            ts = getattr(model, 'rec_'+var)
            plt.plot(model.t[inds], ts)
        plt.legend(rec_vars)
        plt.title("eta_C={} sigma_C={}".format(model.eta_C, model.sigma_C))
        plt.show()


def plot_bold(model, labels=['OFC', 'PFC', 'NAcc', 'Putamen'], colors=['blue', 'green', 'red', 'magenta']):
    """ plot BOLD timeseries and FC """
    fig = plt.figure(figsize=[16,4])
    gs = plt.GridSpec(1,2, width_ratios=[4,1])

    ax1 = fig.add_subplot(gs[0,0])
    for i in range(model.N):
        ts = model.bold_ts.T[:,i]
        ts = (ts-np.mean(ts)) / np.std(ts) - 3*i
        ax1.plot(model.bold_t, ts, color=colors[i], alpha=0.3)
    ax1.set_yticks(-np.linspace(0,4*model.N, model.N)[::-1]) #([-12, -8, -4, 0])
    ax1.set_yticklabels(labels[::-1])
    ax1.legend(labels, loc='upper right')
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Normalized BOLD signal')

    ax2 = fig.add_subplot(gs[0,1])
    img = ax2.imshow(model.bold_fc, vmin=-1, vmax=1, cmap='RdBu_r')
    #plt.colorbar(img)
    plt.xticks(np.arange(len(labels)), labels, rotation=60)
    plt.yticks(np.arange(len(labels)), labels)
    ax2.set_title('Functional Connectivity')
    plt.show()


def plot_correlations(rww, t_range=None):
    """ Visualize correlation between timeseries generated by model (S_rec, not BOLD).
    
    :param rww: ReducedWangWang object.
                
    """
    plt.figure(figsize=[4,4])
    inds = get_inds(rww, t_range)
    corr = np.corrcoef(rww.S_rec[inds,:].T)
    plt.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.xticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    plt.yticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    #plt.colorbar()
    plt.show()
