####    OCD Modeling: Symbolic Reduced Wong-Wang Model Analysis 
###      
##      Author: Sebastien Naze
#       QIMR 2023

from abc import ABC
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
import copy
import datetime
import importlib
import itertools
import joblib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import multiprocessing
import mpmath
import os
import pandas as pd
import pickle
import PyDSTool as dst
import scipy
import sympy as sp
import time

from OCD_modeling.utils import utils
from OCD_modeling.models import ReducedWongWang as RWW

#sp.init_session()
#sp.init_printing()

proj_dir = os.path.join(utils.get_working_dir(), 'lab_lucac/sebastiN/projects/OCD_modeling')

params = {'a':270, 'b': 108, 'd': 0.154, 'C_12':1, 'C_21':1, 'G':2.5, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641}

# increase numerical precision when using mpmath
mpmath.mp.dps = 125

class SymbolicModel(ABC):
    """ Abstract class that implement common functions to all symbolic Reduced Wong Wang models """

    def print_dS(self):
        sp.pprint(self.dS)
        print(sp.latex(self.dS))

    def compute_nullclines(self):
        # using SOLVE
        # -----------
        # nullclines
        self.n1 = sp.solve(self.dS[0], self.S2)
        self.n2 = sp.solve(self.dS[1], self.S1)
        return self.n1,self.n2

    def compute_characteristic_eq(self):
        # substitute S1 into S2 nullcline equation (characteristic equation)
        if (not hasattr(self, 'n1') or not hasattr(self, 'n2')):
            n1,n2 = self.compute_nullclines()
        self.charEq = self.dS[0].subs({self.S1:self.n2[0]})
        return self.charEq

    def plot_nullclines(self, params):
        # plot nullclines and characteritic equation
        sp.plot(self.n1[0].subs(params), (self.S1,-3,3), ylim=(-3,3), n=1000, title='dS_1=0 curve\n')
        sp.plot(self.n2[0].subs(params), (self.S2, 3,3), ylim=(-3,3), n=1000, title='dS_2=0 curve\n')
        sp.plot(self.charEq.subs(params), (self.S2, -3, 3), ylim=(-3,3), n=1000, title='dS=0 curve\n')    


class symRWW_2D(SymbolicModel):
    """ Original Reduced Wong Wang model """
    def __init__(self):

        # model definition
        self.x1, self.x2,self.S1,self.S2,self.C_12,self.C_21 = sp.symbols('x1 x2 S1 S2 C_12 C_21')
        self.S = sp.Matrix([self.S1, self.S2])
        self.x = sp.Matrix([self.x1, self.x2])
        self.C = sp.Matrix([[0, self.C_12],[self.C_21, 0]])

        self.w, self.J_N, self.I_0, self.G = sp.symbols('w J_N I_0 G')
        self.X = self.w*self.J_N*self.S + self.G*self.J_N*self.C*self.S + sp.ones(2,1)*self.I_0

        self.a,self.b,self.d, = sp.symbols('a b d')
        self.H = sp.Matrix([a1/a2 for a1,a2 in zip((self.a*self.X-self.b*sp.ones(2,1)),
                                    sp.ones(2,1)-(-self.d*(self.a*self.X-self.b*sp.ones(2,1))).applyfunc(sp.exp))])

        self.tau_S, self.gamma = sp.symbols('tau_S gamma')
        self.dS = (-self.S/self.tau_S) + sp.matrices.dense.matrix_multiply_elementwise((sp.ones(2,1)-self.S), self.gamma*self.H)
        self.jacobian = [[self.dS[0].diff(self.S1), self.dS[0].diff(self.S2)], [self.dS[1].diff(self.S1), self.dS[1].diff(self.S2)]]


class pw_RWW_2D(SymbolicModel):
    """ Piecewise linear Reduced Wong Wang symbolic model """
    def __init__(self):

        # model definition
        self.x1, self.x2,self.S1,self.S2,self.C_12,self.C_21 = sp.symbols('x1 x2 S1 S2 C_12 C_21')
        self.S = sp.Matrix([self.S1, self.S2])
        self.x = sp.Matrix([self.x1, self.x2])
        self.C = sp.Matrix([[0, self.C_12],[self.C_21, 0]])

        self.w, self.J_N, self.I_0, self.G = sp.symbols('w J_N I_0 G')
        self.X = self.w*self.J_N*self.S + self.G*self.J_N*self.C@self.S + sp.ones(2,1)*self.I_0

        self.a,self.b,self.d,self.theta  = sp.symbols('a b d theta')
        self.H = sp.Matrix([sp.Piecewise((0, self.X[0] <= self.theta), (self.a*(self.X[0]-self.theta), self.X[0]>self.theta)), 
                        sp.Piecewise((0, self.X[1] <= self.theta), (self.a*(self.X[1]-self.theta), self.X[1]>self.theta))])

        self.tau_S, self.gamma = sp.symbols('tau_S gamma')
        self.dS = (-self.S/self.tau_S) + sp.matrices.dense.matrix_multiply_elementwise((sp.ones(2,1)-self.S), self.gamma*self.H)
        self.jacobian = [[self.dS[0].diff(self.S1), self.dS[0].diff(self.S2)], [self.dS[1].diff(self.S1), self.dS[1].diff(self.S2)]]



## Stability analysis
#
def find_roots(f,x,itv=None, slope_thr=0.05):
    """ find zero crossings of function f (numerical roots) 
            input slope_thr defint the slopes above which the zero crossing is an artifact
            e.g. in case of hyperbolic function 
    """
    if itv==None:
        itv = [x.min(), x.max()]
    roots = []
    fxi = np.real(f(x[0]))
    for i,x_i in enumerate(x[1:]):
        #fxii = np.real(complex(f(x[i+1]).evalf()))
        fxii = np.real(f(x_i))
        if ( ((fxi > 0) & (fxii < 0)) | ((fxi < 0) & (fxii > 0)) ):
            diff = fxii - fxi 
            if ((diff < 0) & (np.abs(diff) < slope_thr)):
                roots.append({'x':(x_i+x[i+1])/2, 'slope':diff})
            if ((diff > 0) & (np.abs(diff) < slope_thr)):
                roots.append({'x':(x_i+x[i+1])/2, 'slope':diff})
        fxi = copy.deepcopy(fxii)
    return roots

def find_roots_subs(model, x, default_params):
    """ find roots of equation numerically using sympy substitution instead of lambdified """
    def f(x):
        default_params['S2'] = x
        return model.charEq.subs(default_params)

    roots = find_roots(f,x)
    return roots


def get_fixed_point_stability(model, fp, params):
    """ Derive stability based on eigenvalues of the jacobian around the fixed point """
    fp['tau'] = sp.trace(sp.Matrix(model.jacobian).subs(params))
    fp['delta'] = sp.det(sp.Matrix(model.jacobian).subs(params))
    fp['lambda1'] = (fp['tau'] - sp.sqrt(fp['tau']**2 - 4*fp['delta']))/2
    fp['lambda2'] = (fp['tau'] + sp.sqrt(fp['tau']**2 - 4*fp['delta']))/2
    
    l1_re, l1_im = fp['lambda1'].as_real_imag()
    l2_re, l2_im = fp['lambda2'].as_real_imag()
    
    # handle case of division per 0
    if ((fp['tau']==sp.nan) | (fp['delta']==sp.nan)):
        fp['type'] = None
    # special cases
    elif fp['tau']==0:
        fp['type'] = 'center'
    elif ((fp['tau']**2 - 4*fp['delta'])==0):
        if fp['tau'] > 0:
            fp['type'] = 'degenerate source'
        elif fp['tau'] < 0:
            fp['type'] = 'degenerate sink'
        else: 
            fp['type'] = 'uniform motion'
    # real eigenvalues
    elif ((l1_im==0) & (l2_im==0)):
        # saddle
        if (((l1_re>0) & (l2_re<0)) | ((l1_re<0) & (l2_re>0))):
            fp['type'] = 'saddle'
        # unstable
        elif ((l1_re>0) & (l2_re>0)):
            fp['type'] = 'unstable node'
        elif ((l1_re<0) & (l2_re<0)):
            fp['type'] = 'stable node'
    # complex eigenvalues
    else:
        if ((l1_re>0) & (l2_re>0)):
            fp['type'] = 'unstable focus'
        elif ((l1_re<0) & (l2_re<0)):
            fp['type'] = 'stable focus'


def perform_stability_analysis(model, order_params, default_params, out_queue, x=np.linspace(-3,3,599)):
    """ Analyses the stability of the system of ODEs 
            inputs:
                model : sympy model
                order_params: dictionary of order parameters
                default_params: dictionary of other default parameters    
                x: substitution variable values 
    """
    for k,v in order_params.items(): 
        default_params[k] = v 
    lambdified = sp.lambdify(model.S2, model.charEq.subs(default_params), modules=['mpmath'])
    fps = find_roots(lambdified, x)
    #fps = find_roots_subs(model, x, default_params)
    for fp in fps: 
        fp['S2'] = fp['x']
        default_params['S2'] = fp['S2']
        fp['S1'] = model.n2[0].subs(default_params)
        default_params['S1'] = fp['S1']
        get_fixed_point_stability(model, fp, default_params)
    output = copy.deepcopy(order_params) 
    output['fps'] = fps
    out_queue.put(output)


def launch_stability_analysis(model, order_params, default_params, out_queue, args):
    """ Ghost process that launches the stability analysis for a set of defined order parameter,
        creating a child process with a set timeout per child process """

    proc = multiprocessing.Process(target=perform_stability_analysis, args=(model, order_params, default_params, out_queue))
    proc.start()
    # wait for the process until timeout
    proc.join(args.timeout)
    # if process is still running after timeout, force terminating it 
    if proc.is_alive():
        print(f"Stability analysis for {order_params} took too long, likely a numerical issue, aborted after {args.timeout}s")
        proc.terminate()



def run_stability_analysis(model, order_params, default_params, args):
    # lambdify characterstic equation based on variables of interest
    #charPars = dict((k,v) for k,v in default_params.items() if k not in order_params.keys())
    #variables = (model.S2, *(getattr(model,var) for var in order_params.keys()))
    #lambdified = sp.lambdify(variables, model.charEq.subs(charPars), modules=['mpmath'])
    
    # debug
    #outputs = []
    #for vals in itertools.product(*order_params.values()):
    #    out = perform_stability_analysis(copy.deepcopy(model), dict(zip(order_params.keys(), vals)), copy.deepcopy(default_params))
    #    outputs.append(out)

    #outputs = joblib.Parallel(n_jobs=32, verbose=10, timeout=20)(joblib.delayed(perform_stability_analysis)(copy.deepcopy(model), dict(zip(order_params.keys(), vals)), copy.deepcopy(default_params)) 
    #        for vals in itertools.product(*order_params.values()))

    o_pars = list([dict(zip(order_params.keys(), vals)) for vals in itertools.product(*order_params.values())])
    n_pars = len(o_pars)

    print("Run stability analysis...")
    out_queue = multiprocessing.Queue(maxsize=n_pars)
    futures = []
    with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
        for order_param in o_pars:
            future = pool.submit(launch_stability_analysis, copy.deepcopy(model), copy.deepcopy(order_param), copy.deepcopy(default_params), out_queue, args)
            futures.append(future)
    #print(futures)
    outputs = []
    while not out_queue.empty():
        outputs.append(out_queue.get_nowait())
    return outputs, futures



def plot_3d_bifurcations(outputs, azim=0, elev=0):
    """ plot bifurcation diagram in 3D """
    node_colors = {'saddle': 'purple', 'unstable node': 'red', 'stable node':'blue', 'unstable focus':'magenta', 'stable focus': 'green',
                    'degenerate source':'black', 'degenerate sink':'black', 'uniform motion':'black'}

    fig = plt.figure(figsize=[20,8])

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.view_init(azim=azim,elev=elev)

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.view_init(azim=azim,elev=elev)

    for output in outputs:
        o_pars = np.sort([k for k in output.keys() if k!='fps'])
        for fp in output['fps']:
            if fp['type']!=None:
                marker = matplotlib.markers.MarkerStyle('o', fillstyle='full')
                opts = {'marker':marker, 'color':node_colors[fp['type']], 'alpha':0.2}
                ax1.scatter(output[o_pars[0]], output[o_pars[1]], fp['S1'], **opts)
                ax2.scatter(output[o_pars[0]], output[o_pars[1]], fp['S2'], **opts)


        ax1.set_xlim(-1,1)
        ax1.set_ylim(-1,1)
        ax1.set_xlabel(o_pars[0])
        ax1.set_ylabel(o_pars[1])

        ax2.set_xlim(-1,1)
        ax2.set_ylim(-1,1)
        ax2.set_xlabel(o_pars[0])
        ax2.set_ylabel(o_pars[1])

    plt.show()


def get_model(args):
    """ create or load symbolic model """
    if args.create_model:
        print("Creating model..")
        #sym_rww = symRWW_2D()
        sym_rww = pw_RWW_2D()
        if args.compute_nullclines:
            print("Computing nullclines.. (takes a few minutes)")
            t0 = time.time()
            sym_rww.compute_characteristic_eq()
            print('Done in {:0.1f}s'.format(time.time()-t0))
        if args.save_model:
            print("Saving model...")
            with open(os.path.join(proj_dir, 'postprocessing', 'sym_pw_rww.pkl'), 'wb') as f:
                pickle.dump(sym_rww, f)
    else:
        with open(os.path.join(proj_dir, 'postprocessing', 'sym_rww.pkl'), 'rb') as f:
            sym_rww = pickle.load(f)
    return sym_rww


def parse_args():
    """ parsing global argument """
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_model', default=False, action='store_true', help='create symbolic model')
    parser.add_argument('--compute_nullclines', default=False, action='store_true', help='create symbolic model')
    parser.add_argument('--save_model', default=False, action='store_true', help='save symbolic model with its ciomputed attributes')
    parser.add_argument('--run_stability_analysis', default=False, action='store_true', help='run stability analysis: find fixed point (semi-analytically) and perform linear stability analysis around them')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save analysis outputs')
    parser.add_argument('--timeout', type=int, default=30, action='store', help='timeout of the stability analysis (per parameter combination invoked)')
    parser.add_argument('--n_jobs', type=int, default=12, action='store', help='numper of processes used in parallelization')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    sym_rww = get_model(args)
    order_params = {'C_12': np.arange(-1,1,0.025), 'C_21': np.arange(-1,1,0.2)}
    if args.run_stability_analysis:
        outputs, futures = run_stability_analysis(sym_rww, order_params, params, args)
        if args.save_outputs:
            today = datetime.datetime.now().strftime("%Y%m%d")
            fname = 'outputs_'+today+'.pkl'
            with open(os.path.join(proj_dir, 'postprocessing', fname), 'wb') as f:
                pickle.dump(outputs, f)
        if args.plot_figs:
            plot_3d_bifurcations(outputs, azim=0, elev=0) # yz plan
            plot_3d_bifurcations(outputs, -90, 0) # xz plan
            plot_3d_bifurcations(outputs, 90, -90) # xy plan
