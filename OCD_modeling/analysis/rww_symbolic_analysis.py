####    OCD Modeling: Reduced Wong-Wang Model Analysis 
###      
##      Author: Sebastien Naze
#       QIMR 2023

import argparse 
import importlib
import itertools
import joblib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import scipy
import sympy as sp

from OCD_modeling.models import ReducedWongWang as RWW
importlib.reload(RWW)

sp.init_session()
sp.init_printing()

params = {'a':270, 'b': 108, 'd': 0.154, 'C_12':1, 'C_21':1, 'G':1, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641}

# model definition
x1,x2,S1,S2,C_12,C_21 = sp.symbols('x1 x2 S1 S2 C_12 C_21')
S = sp.Matrix([S1, S2])
x = sp.Matrix([x1, x2])
C = sp.Matrix([[0, C_12],[C_21, 0]])

w, J_N, I_0, G = sp.symbols('w J_N I_0 G')
X = w*J_N*S + G*J_N*C*S + sp.ones(2,1)*I_0

a,b,d, x_i = sp.symbols('a b d x_i')
H = sp.Matrix([a1/a2 for a1,a2 in zip((a*X-b*sp.ones(2,1)),sp.ones(2,1)-(-d*(a*X-b*sp.ones(2,1))).applyfunc(sp.exp))])

tau_S, gamma = sp.symbols('tau_S gamma')
dS = (-S/tau_S) + sp.matrices.dense.matrix_multiply_elementwise((sp.ones(2,1)-S), gamma*H)
sp.pprint(dS)
print(sp.latex(dS))

# using SOLVE
# -----------
# nullclines
n1 = sp.solve(dS[0], S2)
n2 = sp.solve(dS[1], S1)

# substitute S1 into S2 nullcline equation (characteristic equation)
n0 = dS[0].subs({S1:n2[0]})

# plot nullclines and characteritic equation
sp.plot(n1[0].subs(params), (S1,-3,3), ylim=(-3,3), n=1000, title='dS_1=0 curve\n')
sp.plot(n2[0].subs(params), (S2, 3,3), ylim=(-3,3), n=1000, title='dS_2=0 curve\n')
sp.plot(n0.subs(params), (S2, -3, 3), ylim=(-3,3), n=1000, title='dS=0 curve\n')    



## Stability analysis
#
def find_roots(f,x,itv=None, slope_thr=1):
    """ find zero crossings of function f (numerical roots) """
    if itv==None:
        itv = [x.min(), x.max()]
    roots = []
    for i,x_i in enumerate(x[:-1]):
        if ( ((f(x_i) > 0) & (f(x[i+1]) < 0)) | ((f(x_i) < 0) & (f(x[i+1]) > 0)) ):
            diff = f(x[i+1]) - f(x_i) 
            if ((diff < 0) & (np.abs(diff) < slope_thr)):
                roots.append({'x':(x_i+x[i+1])/2, 'slope':diff})
            if ((diff > 0) & (np.abs(diff) < slope_thr)):
                roots.append({'x':(x_i+x[i+1])/2, 'slope':diff})
    return roots

jacobian = [[dS[0].diff(S1), dS[0].diff(S2)], [dS[1].diff(S1), dS[1].diff(S2)]]


def get_fixed_point_stability(fp, params):
    """ Derive stability based on eigenvalues of the jacobian around the fixed point """
    fp['tau'] = sp.trace(sp.Matrix(jacobian).subs(params))
    fp['delta'] = sp.det(sp.Matrix(jacobian).subs(params))
    fp['lambda1'] = (fp['tau'] - sp.sqrt(fp['tau']**2 - 4*fp['delta']))/2
    fp['lambda2'] = (fp['tau'] + sp.sqrt(fp['tau']**2 - 4*fp['delta']))/2
    
    l1_re, l1_im = fp['lambda1'].as_real_imag()
    l2_re, l2_im = fp['lambda2'].as_real_imag()
    
    # handle case of division per 0
    if ((fp['tau']==sp.nan) | (fp['delta']==sp.nan)):
        fp['type'] = None
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
    return fp

def perform_stability_analysis(dS, ncs, order_params, default_params):
    """ Analyses the stability of the system of ODEs 
            inputs:
                dS : differential equations (sympy Matrix equation)
                ncs: nullclines (list of sympy equations)
                order_params: dictionary of order parameters
                default_params: dictionary of other default parameters    

    """
    for k,v in order_params.items(): 
        default_params[k] = v
    g = sp.lambdify(S2, dS[1].subs({'S1':ncs[1]}).subs(default_params))
    fps = find_roots(g,s)
    for fp in fps: 
        fp['S2'] = fp['x']
        default_params['S2'] = fp['S2']
        fp['S1'] = ncs[1].subs(default_params)
        default_params['S1'] = fp['S1']
        fp = get_fixed_point_stability(fp, default_params)
    order_params['fps'] = fps
    return order_params


params = {'a':270, 'b': 108, 'd': 0.154, 'C_12':1, 'C_21':1, 'G':1, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641}
s = np.linspace(-3,3,999)
order_params = {'C_12': np.arange(-1,1,0.025), 'C_21': np.arange(-1,1,0.025)}
outputs = joblib.Parallel(n_jobs=32)(joblib.delayed(perform_stability_analysis)(dS,[n1[0],n2[0]], dict(zip(order_params.keys(), vals)), params.copy()) 
        for vals in itertools.product(*order_params.values()))

# debug
#outputs = []
#for vals in itertools.product(*order_params.values()):
#    out = perform_stability_analysis(dS,[n1[0],n2[0]], dict(zip(order_params.keys(), vals)), params.copy())
#    outputs.append(out)



def plot_3d_bifurcations(outputs):
    """ plot bifurcation diagram in 3D """
    node_colors = {'saddle': 'purple', 'unstable node': 'red', 'stable node':'blue', 'unstable focus':'magenta', 'stable focus': 'green'}

    fig = plt.figure(figsize=[20,8])

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.view_init(elev=30,azim=45)

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.view_init(30,45)

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


plot_3d_bifurcations(outputs)

"""
def main():
    analyse_model(args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    main(args)
"""