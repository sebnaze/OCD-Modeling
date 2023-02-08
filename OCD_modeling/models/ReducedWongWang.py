import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import pickle
import platform
import scipy
import scipy.stats
import sklearn
from sklearn import preprocessing

from OCD_modeling.models.HemodynamicResponseModeling.BalloonWindkessel import balloonWindkessel
from OCD_modeling.utils.neurolib.neurolib.models.bold.timeIntegration import simulateBOLD

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
        self.sigma = sigma     # noise amplitude (nA); default=0.001
        self.v_i = v_i         # gaussian noise (n/a); default=0

    def H(self, x):
        """ Average synaptic gating
            -----------------------
                a: slope (n/C); default=270
                b: offset (Hz); default=108
                d: decay (s);   default=0.154
        """

        return (self.a * x - self.b) / (1. - np.exp(-self.d * (self.a * x - self.b)))

    def S_i(self, x, S_j=0):
        """ Firing rate
            -----------
                I_0: external input (nA); default=0.3
                J_N: synaptic coupling (nA); default=0.2609
                w: local exc recurrence (n/a); default=0.9
                G: global scaling factor (n/a); default=1
                C: connectivity matrix (n/a); default=0
                S_j: coupled population firing rates (Hz); default=0
        """
        return (x - self.I_0 - self.G * self.J_N * self.C * S_j) / (self.w * self.J_N)

    def dS_i(self, x, v_i=0):
        """ ODE of firing rate
            ------------------
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
                 C=None, S=None, x=None):

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
            self.S = np.random.rand(N,)         # coupled population firing rates (Hz);
        else:
            self.S = np.array(S).squeeze().astype(float)
        self.x = np.random.rand(N,)         # coupled population activity (Hz); default = uniformly distributed

        # ODE params
        self.tau_S = tau_S      # kinetic parameter of local population (ms); default=100
        self.gamma = gamma     # kinetic parameter of coupled population (ms); default=0.000641
        self.sigma = sigma     # noise amplitude (nA); default=0.001
        #self.v = v_0         # gaussian noise (n/a); default=0
        self.dt = dt
        self.N = N          # number of units

        if C is None:
            self.C = np.random.randn(N,N) * (1-np.eye(N,N))  # connectivity matrix (n/a); default= 0 self, 1 to others
        else:
            self.C = C

    def H(self, x):
        """ Average synaptic gating
            -----------------------
                a: slope (n/C); default=270
                b: offset (Hz); default=108
                d: decay (s);   default=0.154
        """

        return (self.a * x - self.b) / (1. - np.exp(-self.d * (self.a * x - self.b)))

    def v(self):
        """ implementing the wiener process """
        return np.random.randn(self.N,)

    def dS(self):
        """ ODE of firing rate
            ------------------
                tau_S: kinetic parameter of local population (ms); default=100
                gamma: kinetic parameter of coupled population (ms); default=0.000641
                sigma: noise amplitude (nA); default=0.001
                v_i: gaussian noise (n/a); default=0
        """
        return (-self.S/self.tau_S + (1 - self.S) @ (self.gamma * self.H(x=self.x)) + self.sigma*self.v())

    """def x(self):
        return self.w * self.J_N * self.S + self.G * self.J_N * self.C @ self.S + self.I_0
    """

    def integrate(self):
        """ Euler integration of the ODE """
        self.x = self.w * self.J_N * self.S + self.G * self.J_N * self.C @ self.S + self.I_0
        H = self.H(x=self.x)
        v = self.v()
        dS = (-self.S/self.tau_S + (1 - self.S) * (self.gamma * H) + self.sigma*v)
        self.S = self.S + dS*self.dt

    def run(self, t_tot=1000, sf=100, t_rec=None):
        """ Run the model
                t_tot:  total simu;ation time (s)
                sf:     sampling frequency of the reccording (Hz)
                t_rec: interval of recording (s)
        """
        n_ts = int(t_tot/self.dt)
        sf_dt = 1./(sf*self.dt)

        # fix sf if needed
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

        if t_rec==None:
            t_rec = [0,t_tot]
        n_rec = int((t_rec[1]-t_rec[0])*self.sf)  # number of steps recorded
        self.S_rec = np.zeros((n_rec,self.N))
        self.t = np.arange(t_rec[0],t_rec[1], 1./self.sf)

        # run the model
        rec_idx = 0
        for i in range(n_ts):
            self.integrate()
            # record
            if ((not i%sf_dt) & ((i*self.dt)>=t_rec[0]) & ((i*self.dt)<=t_rec[1])):
                self.S_rec[rec_idx,:] = self.S.copy()
                rec_idx += 1


## FUNCTIONS ##
# ----------- #
def score_model(rww, coh='con'):
    """ score model against empirical FC
            rww:  instance of model to score
            coh:    cohort to be scored against ('con' or 'pat')
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
    corr_diff = np.sum(np.abs(corrData - corrModel))
    r,pval = scipy.stats.pearsonr(corrData, corrModel)
    return r, corr_diff, corr

def distance(x,y):
    """ distance to minimize based on score """
    return 1 - x['r'] + x['corr_diff']

def get_inds(model, t_range=None):
    """ extract time series indices of interest beased on t_range (in sec) """
    if t_range==None:
        t_range=[model.t.min(), model.t.max()]
    inds, = np.where((model.t>=t_range[0]) & (model.t<=t_range[1]))
    return inds

def plot_timeseries(rww, t_range=None):
    """ visualize time serie generated by model
            intputs:
                rww: ReducedWangWang object"""
    plt.figure(figsize=[16,4])
    inds = get_inds(rww, t_range)
    plt.plot(rww.t[inds],rww.S_rec[inds,:])
    #plt.legend([str(i) for i in range(rww.N)])
    plt.legend(['OFC', 'PFC', 'NAcc', 'Put'])
    plt.show()

def compute_bold(model, t_range=None):
    """ BOLD timeseries and functional connectivity between regions """
    inds = get_inds(model, t_range)
    #bold_ts, s, f, v, q = balloonWindkessel(model.S_rec[inds,:].T, 1./model.sf)
    scaler = sklearn.preprocessing.MinMaxScaler()
    ts = scaler.fit_transform(model.S_rec[inds,:])
    bold_ts, x, f, q, v = simulateBOLD(ts.T, 1./model.sf, voxelCounts=None)
    model.bold_ts = bold_ts[:,int(model.sf)*10:] # discard first 10 sec due to transient
    model.bold_fc = np.corrcoef(model.bold_ts)

def plot_bold(model):
    """ plot BOLD timeseries and FC """
    fig = plt.figure(figsize=[16,4])
    gs = plt.GridSpec(1,2, width_ratios=[4,1])

    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(model.bold_ts.T)
    ax1.legend(['OFC', 'PFC', 'NAcc', 'Put'])

    ax2 = fig.add_subplot(gs[0,1])
    ax2.imshow(model.bold_fc, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.xticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    plt.yticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    plt.show()


def plot_correlations(rww, t_range=None):
    """ visualize correlation between timeseries generated by model (S_rec, not BOLD)
            intputs:
                rww: ReducedWangWang object"""
    plt.figure(figsize=[4,4])
    inds = get_inds(rww, t_range)
    corr = np.corrcoef(rww.S_rec[inds,:].T)
    plt.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.xticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    plt.yticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    #plt.colorbar()
    plt.show()
