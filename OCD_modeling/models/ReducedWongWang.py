import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import pickle
import platform
import scipy
import scipy.stats


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
    """ Reduced Wong Wang model """
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












class ReducedWongWang2D:
    """ Reduced Wong Wang model """
    def __init__(self, a=270., b=108., d=0.154,
                 I_0=0.3, J_N=0.2609, w=0.9, G=1.,
                 tau_S=100., gamma=0.000641, sigma=0.001, N=2, dt=0.001,
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
            self.S = (np.random.rand(N,)-0.5)*4         # coupled population firing rates (Hz);
        else:
            self.S = np.array(S).squeeze().astype(float)
        self.x = (np.random.rand(N,)-0.5)*4         # coupled population activity (Hz); default = uniformly distributed

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

    def run(self, t_tot=100, sf=1000):
        """ Run the model
                t_tot:  total simu;ation time (s)
                sf:     sampling frequency of the reccording (Hz)
        """
        n_ts = int(t_tot/self.dt)
        sf_dt = 1./(sf*self.dt)

        # fix sf if needed
        if not sf_dt.is_integer():
            if sf_dt<1:
                sf_dt=1
            else:
                sf_dt = np.floor(sf_dt)
            self.sf = sf_dt / self.dt
            print("Sampling frequency not a multiple of dt, used sf={} instead".format(sf))
        self.S_rec = np.zeros((n_ts,self.N))
        self.t = np.arange(0,t_tot, sf_dt*self.dt)

        # run the model
        for i in range(n_ts):
            self.integrate()
            # record
            if not i%sf_dt:
                self.S_rec[int(i/sf_dt),:] = self.S.copy()

def score_model(rww, t_range=None):
    """ score model against empirical FC  """
    # load empirical FC
    with open(os.path.join(proj_dir, 'postprocessing', 'R.pkl'), 'rb') as f:
        R = pickle.load(f)
    # compute model FC
    if t_range==None:
        t_range=[rww.t.min(), rww.t.max()]
    inds, = np.where((rww.t>t_range[0]) & (rww.t<t_range[1]))
    ts = rww.S_rec[inds,:]
    corr = np.corrcoef(ts.T)
    # compute score
    score = dict()
    fix_inds = [2,3,0,1] # NAcc Put OFC PFC -> OFC PFC NAcc Put
    fixed_inds = np.ix_(fix_inds,fix_inds)
    triu_inds = np.triu_indices(ts.shape[1],k=1)
    score['con'] = scipy.stats.pearsonr(R['con'][fixed_inds][triu_inds].flatten(), corr[triu_inds].flatten())
    score['pat'] = scipy.stats.pearsonr(R['pat'][fixed_inds][triu_inds].flatten(), corr[triu_inds].flatten())
    score['corr_matrix'] = corr
    return score


def plot_timeseries(rww, t_range=None):
    """ visualize time serie generated by model
            intputs:
                rww: ReducedWangWang object"""
    plt.figure(figsize=[16,4])
    if t_range==None:
        t_range=[rww.t.min(), rww.t.max()]
    inds, = np.where((rww.t>t_range[0]) & (rww.t<t_range[1]))
    plt.plot(rww.t[inds],rww.S_rec[inds,:])
    #plt.legend([str(i) for i in range(rww.N)])
    plt.legend(['OFC', 'PFC', 'NAcc', 'Put'])
    plt.show()


def plot_correlations(rww, t_range=None):
    """ visualize correlation between timeseries generated by model
            intputs:
                rww: ReducedWangWang object"""
    plt.figure(figsize=[4,4])
    if t_range==None:
        t_range=[rww.t.min(), rww.t.max()]
    inds, = np.where((rww.t>t_range[0]) & (rww.t<t_range[1]))
    ts = rww.S_rec[inds,:]
    corr = np.corrcoef(ts.T)
    plt.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.xticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    plt.yticks([0,1,2,3], ['OFC', 'PFC', 'NAcc', 'Put'])
    #plt.colorbar()
    plt.show()
