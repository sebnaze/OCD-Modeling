####    OCD Modeling: Reduced Wong-Wang Model Analysis using Dynamical Systems Toolbox
###      
##      Author: Sebastien Naze
#       QIMR 2023

from abc import ABC
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
import copy
import dill
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
from pathos.multiprocessing import ProcessingPool
import pdb
import pickle
import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp
import scipy
import sympy as sp
import time


from OCD_modeling.utils.utils import *
from OCD_modeling.models import ReducedWongWang as RWW



rng = np.random.default_rng()

dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump
multiprocessing.queues._ForkingPickler = dill.Pickler

def create_model(params, args):
    """ Create the Dynamical System in PyDSTool """
    icdict = {'S1': rng.uniform(),
            'S2': rng.uniform()}
    S1eq = '-S1/tau_S + (1-S1)*gam*(a*(w*J_N*S1+G*J_N*C_12*S2+I_0-I_1)-b)/(1-exp(-d*(a*(w*J_N*S1+G*J_N*C_12*S2+I_0-I_1)-b)))'
    S2eq = '-S2/tau_S + (1-S2)*gam*(a*(w*J_N*S2+G*J_N*C_21*S1+I_0+I_1)-b)/(1-exp(-d*(a*(w*J_N*S2+G*J_N*C_21*S1+I_0+I_1)-b)))'

    DSargs = dst.args(name='rww')  # struct-like data
    DSargs.pars = params
    DSargs.tdata = [0, 100]
    DSargs.algparams = {'max_pts': 100000, 'init_step': 0.01, 'stiff': True}
    DSargs.varspecs = {'S1': S1eq, 'S2': S2eq}
    DSargs.xdomain = {'S1': [-1, 1], 'S2': [-1, 1]}
    
    DSargs.fnspecs = {'Jacobian': (['t','S1','S2'],
                                    """[[-J_N*a*d*gam*w*(1 - S1)*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b)*exp(-d*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b))/(1 - exp(-d*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b)))**2 + J_N*a*gam*w*(1 - S1)/(1 - exp(-d*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b))) - gam*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b)/(1 - exp(-d*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b))) - 1/tau_S, -C_12*G*J_N*a*d*gam*(1 - S1)*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b)*exp(-d*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b))/(1 - exp(-d*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b)))**2 + C_12*G*J_N*a*gam*(1 - S1)/(1 - exp(-d*(a*(C_12*G*J_N*S2 + I_0 - I_1 + J_N*S1*w) - b)))],
                                        [-C_21*G*J_N*a*d*gam*(1 - S2)*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b)*exp(-d*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b))/(1 - exp(-d*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b)))**2 + C_21*G*J_N*a*gam*(1 - S2)/(1 - exp(-d*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b))), -J_N*a*d*gam*w*(1 - S2)*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b)*exp(-d*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b))/(1 - exp(-d*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b)))**2 + J_N*a*gam*w*(1 - S2)/(1 - exp(-d*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b))) - gam*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b)/(1 - exp(-d*(a*(C_21*G*J_N*S1 + I_0 + I_1 + J_N*S2*w) - b))) - 1/tau_S]]""")}

    DSargs.ics = icdict
    rww = dst.Vode_ODEsystem(DSargs)
    return rww


def get_fixed_points(model, params, xdomain={'S1':[0,1], 'S2':[0,1]}, args=None):
    """ Get model's fixed point and nullclines for a given set of parameters """
    model.set(pars=params, xdomain=xdomain)
    # fixed points (using n starting points along the domain)
    fp_coords = pp.find_fixedpoints(model, n=20, eps=1e-8) 
    fps = []
    for fp_coord in fp_coords:
        fp = pp.fixedpoint_2D(model, dst.Point(fp_coord), eps=1e-8)
        fps.append(fp)
    # nullclines (using n starting points along the domain)
    nulls_x, nulls_y = pp.find_nullclines(model, 'S1', 'S2', n=5, eps=1e-8, max_step=0.01, fps=fp_coords)
    return model, fps, (nulls_x, nulls_y)


def compute_tajectories(model, n, tdata=[0,500]):
    """ Compute n trajectories from model, each with different initial conditions, and of time refered in tdata ([start,stop]) """
    points = []
    for i in range(n):
        model.set(ics={'S1':rng.uniform(), 'S2':rng.uniform()},
                  tdata=tdata)
        traj = model.compute(f"traj {i}")
        pts = traj.sample()
        points.append(pts)
    return points



def compute_equilibrium_point_curve(model, fps, pdomain):
    """ find equilibrium point curve(s) of the system, starting from each fixed point (if exist)""" 
    model.set(pdomain=pdomain)
    cont = dst.ContClass(model)
    free_params = list(pdomain.keys())
    for i,fp in enumerate(fps):
        epc = 'EQ'+str(i)
        PCargs = dst.args(name=epc, type='EP-C')
        PCargs.initpoint = fp.point
        PCargs.freepars = free_params
        PCargs.StepSize = 1e-5
        PCargs.MaxNumPoints = 1000000
        PCargs.MaxStepSize = 1e-4
        PCargs.LocBifPoints = 'all'
        PCargs.StopAtPoints = 'B'
        PCargs.SaveEigen = True
        PCargs.verbosity = 0
        cont.newCurve(PCargs)

        print('Computing curve...')
        start = dst.perf_counter()
        cont[epc].forward()
        cont[epc].backward()
        print('done in %.3f seconds!' % (dst.perf_counter()-start))
    return cont



def stability_analysis(order_params, default_params, out_queue, args, pdomain={'C_12':[-0.5, 0.5]}):
    """ Create model and analyse dynamics using PyDSTool """
    for k,v in order_params.items(): 
        default_params[k] = v
    
    out = dict()
    
    # fixed point and nullclines
    model = create_model(default_params, args)
    model, fps, nullclines = get_fixed_points(model, default_params)
    points = compute_tajectories(model, args.n_trajs)
    out['model'], out['fps'], out['ncs'], out['trajs'] = model, fps, nullclines, points

    # EP-C
    if args.compute_epc:
        # takes a bit of ressource and memory
        try:
            cont = compute_equilibrium_point_curve(model, fps, pdomain)
            out['dilled_cont'] = dill.dumps(cont, byref=True)
        except:
            print("Error in computing Equilibrium Point Curve")

    # put output to output queue
    output = copy.deepcopy(order_params)
    output['output'] = out
    out_queue.put(dill.dumps(output))
    



def launch_stability_analysis(order_params, default_params, out_queue, args):
    """ Ghost process that launches the stability analysis for a set of defined order parameter,
        creating a child process with a set timeout per child process """
    proc = multiprocessing.Process(target=stability_analysis, args=(order_params, default_params, out_queue, args))
    proc.start()
    # wait for the process until timeout
    proc.join(args.timeout)
    # if process is still running after timeout, force terminating it 
    if proc.is_alive():
        print(f"Stability analysis for {order_params} took too long, aborted after {args.timeout}s")
        proc.terminate()



def run_stability_analysis(order_params, default_params, args):
    """ Starts a pool of parallel processes to run the stability analysis """
    # debug
    #outputs = []
    #for vals in itertools.product(*order_params.values()):
    #    out = stability_analysis(dict(zip(order_params.keys(), vals)), copy.deepcopy(default_params), out_queue=None, args=args)
    #    outputs.append(out)
    #return outputs, []
    #outputs = joblib.Parallel(n_jobs=32, verbose=10, timeout=20)(joblib.delayed(perform_stability_analysis)(copy.deepcopy(model), dict(zip(order_params.keys(), vals)), copy.deepcopy(default_params)) 
    #        for vals in itertools.product(*order_params.values()))
    
    o_pars = list([dict(zip(order_params.keys(), vals)) for vals in itertools.product(*order_params.values())])
    n_pars = len(o_pars)

    print("Run stability analysis...")
    #out_queue = multiprocessing.Queue(maxsize=n_pars) #  /!\ Standard queues seems (at least) 10x slower than managed queues, resulting in timeouts..  
    out_queue = multiprocessing.Manager().Queue()      #         if performance is primordial, it could be changed to Pipes with locks (even faster)
    futures = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        #future = [pool.map(launch_stability_analysis, (copy.deepcopy(order_param), copy.deepcopy(default_params), out_queue, args)) for order_param in o_pars]
        for order_param in o_pars:
            future = pool.submit(launch_stability_analysis, copy.deepcopy(order_param), copy.deepcopy(default_params), out_queue, args)
            futures.append(future)
    #with ProcessingPool(nodes=args.n_jobs) as pool:
    #    params = [(copy.deepcopy(order_param), copy.deepcopy(default_params), out_queue, args) for order_param in o_pars]
    #    futures = pool.uimap(launch_stability_analysis, params)
    
    outputs = []
    while not out_queue.empty():
        outputs.append(out_queue.get_nowait())
    return outputs, futures
    

def plot_phasespace(model, fps, nullclines, trajs, n=5, eps=1e-8, ax=None, args=None):
    """ Plot vector field, nullclines and fixed points of a model previously set with parameters """
    plt.sca(ax)
    pp.plot_PP_vf(model, 'S1', 'S2', scale_exp=-2)
    pp.plot_PP_fps(fps, do_evecs=True, markersize=5)
    plt.plot(nullclines[0][:,0], nullclines[0][:,1], 'b', lw=2)
    plt.plot(nullclines[1][:,0], nullclines[1][:,1], 'g', lw=2)
    for traj in trajs:
        plt.plot(traj['S1'], traj['S2'], linewidth=1)
    #plt.show(block=False)


def get_grid_inds(output, order_params):
    """ get the indices (i,j) of the order parameters from output """
    o_pars = np.sort([k for k in output.keys() if k!='output'])
    i = [x for x,val in enumerate(order_params[o_pars[0]]) if val==output[o_pars[0]]]
    j = [y for y,val in enumerate(order_params[o_pars[1]]) if val==output[o_pars[1]]]
    return o_pars, i[0],j[0]

def plot_phasespace_grid(outputs, order_params, args=None):
    """ Plot a grid of graphs from outputs """
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.size':11, 'axes.titlesize':'medium', 'mathtext.default': 'regular'})
    fig = plt.figure(figsize=[16,16])
    p1s, p2s = list(order_params.values())[0], list(order_params.values())[1]
    gs = plt.GridSpec(nrows=len(p1s), ncols=len(p2s))
    
    for res in outputs:
        out = dill.loads(res)
        output = out['output']
        o_pars,i,j = get_grid_inds(out, order_params)
        ax = fig.add_subplot(gs[i,j])
        plot_phasespace(output['model'], output['fps'], output['ncs'], output['trajs'], ax=ax, args=args)

        plt.axis('tight')
        #plt.title("{}={:.3f}  {}={:.3f}".format(o_pars[0], p1s[i], o_pars[1], p2s[j]))
        ttl = "$C_{1 \leftarrow 2}$=%s        $C_{2 \leftarrow 1}$=%s" %("{:.2f}".format(p1s[i]),"{:.2f}".format(p2s[j]))
        plt.title(ttl, fontdict={'fontsize': 11} )#.format(p1s[i], p2s[j]))
        
        if i==len(p1s)-1:
            plt.xlabel('$S_1$')
        if j==0:
            plt.ylabel('$S_2$')
        plt.tight_layout()

        # makes quiver a bit more transparent and arrows a bit larger
        plt.getp(ax, 'children')[0].set(alpha=0.7)
        #for path in plt.getp(plt.getp(ax,'children')[0], 'paths'):
        #    path.vertices = path.vertices*10

    #pdb.set_trace()
    if args.save_figs:
        today = datetime.datetime.now().strftime("%Y%m%d")
        plt.savefig(os.path.join(proj_dir, 'img', 'phase_space'+today+'.svg'), format='svg', transparent=True)
    if args.plot_figs:
        plt.show()
    

def plot_bifurcation_grid(outputs, order_params, args=None):
    """ Plot a grid of bifurcation diagrams """
    plt.rcParams.update({'font.size':6, 'axes.titlesize':'medium'})
    fig = plt.figure(figsize=[20,20])
    p1s, p2s = list(order_params.values())[0], list(order_params.values())[1]
    gs = plt.GridSpec(nrows=len(p1s), ncols=len(p2s))
    for output in outputs: 
        o_pars,i,j = get_grid_inds(output, order_params)
        ax = fig.add_subplot(gs[i,j])
        if 'output' in output.keys():
            cont = dill.loads(output['output']['dilled_cont'])
            try:
                cont.display(axes=ax, coords=['C_12', 'S2'], stability=True, color='blue')
                cont.display(axes=ax, coords=['C_12', 'S1'], stability=True, color='orange')
            except:
                plt.xlabel("")
                plt.xticks([])
                plt.ylabel("")
                plt.yticks([])
                plt.title("{}={:.3f}  {}={:.3f}".format(o_pars[0], p1s[i], o_pars[1], p2s[j]))
                continue
            plt.sca(ax)
            plt.axis('tight')
            plt.title("{}={:.3f}  {}={:.3f}".format(o_pars[0], p1s[i], o_pars[1], p2s[j]))
            if i<len(p1s)-1:
                plt.xlabel("")
                plt.xticks([])
            if j > 0:
                plt.ylabel("")
                plt.yticks([])
            plt.xlim([-1,1])
            plt.ylim([0,1])
        else:
            plt.xlabel("")
            plt.xticks([])
            plt.ylabel("")
            plt.yticks([])
            plt.title("{}={:.3f}  {}={:.3f}".format(o_pars[0], p1s[i], o_pars[1], p2s[j]))
    plt.show(block=False)

def plot_bifurcation_row(outputs, order_params, args=None):
    """ Plot a row of bifurcation diagrams (ie. a 1 by n grid) """
    plt.rcParams.update({'font.size':10, 'axes.titlesize':'medium'})
    fig = plt.figure(figsize=[15,3])
    p1s = list(order_params.values())[0]
    gs = plt.GridSpec(nrows=1, ncols=len(p1s))
    for output in outputs:
        output = dill.loads(output) 
        o_par = list(order_params.keys())[0]
        i = [x for x,val in enumerate(order_params[o_par]) if val==output[o_par]][0]
        ax = fig.add_subplot(gs[0,i])
        if 'output' in output.keys():
            cont = dill.loads(output['output']['dilled_cont'])
            try:
                cont.display(axes=ax, coords=['C_12', 'S1'], stability=True, color='blue')
                cont.display(axes=ax, coords=['C_12', 'S2'], stability=True, color='red')
            except:
                plt.xlabel("")
                plt.xticks([])
                plt.ylabel("")
                plt.yticks([])
                plt.title("{}={:.3f}".format(o_par, p1s[i]))
                continue
            plt.sca(ax)
            plt.axis('tight')
            plt.title("{}={:.3f}".format(o_par, p1s[i]))
            if i > 0:
                plt.ylabel("")
                plt.yticks([])
            plt.xlim([-0.5,0.5])
            plt.ylim([0,1])
        else:
            plt.xlabel("")
            plt.xticks([])
            plt.ylabel("")
            plt.yticks([])
            plt.title("{}={:.3f}".format(o_par, p1s[i]))
    plt.tight_layout()
    
    if args.save_figs:
        fname = os.path.join(proj_dir, 'img', 'bifurcation_diagram_023_'+today()+'.svg')
        plt.savefig(fname)
    
    if args.plot_figs:
        plt.show(block=False)
    else:
        fig.close()



def parse_args():
    """ parsing global argument """
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_model', default=False, action='store_true', help='create symbolic model')
    parser.add_argument('--compute_nullclines', default=False, action='store_true', help='compute nullclines numerically')
    parser.add_argument('--compute_epc', default=False, action='store_true', help='compute equilibrium point curves numerically')
    parser.add_argument('--save_model', default=False, action='store_true', help='save symbolic model with its ciomputed attributes')
    parser.add_argument('--run_stability_analysis', default=False, action='store_true', help='run stability analysis: find fixed point (semi-analytically) and perform linear stability analysis around them')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--plot_phasespace_grid', default=False, action='store_true', help='plot grid of phase spaces using order_params')
    parser.add_argument('--plot_bifurcation_diagrams', default=False, action='store_true', help='plot grid of phase spaces using order_params')
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save analysis outputs')
    parser.add_argument('--timeout', type=int, default=60, action='store', help='timeout of the stability analysis (per parameter combination invoked)')
    parser.add_argument('--n_jobs', type=int, default=20, action='store', help='number of processes used in parallelization')
    parser.add_argument('--n_trajs', type=int, default=10, action='store', help='number of trajectories (traces) to compute for phase space projection')
    parser.add_argument('--n_op', type=int, default=5, action='store', help='number of values taken for each order parameters')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    default_params = {'a':270, 'b': 108, 'd': 0.154, 'C_12': 0, 'G':2.5, 'J_N':0.2609, 'I_0':0.3, 'I_1':0.0, 'tau_S':100, 'w':0.9, 'gam':0.000641}
    #order_params = {'C_12': np.linspace(-1,1,args.n_op), 'I_0': np.linspace(0.2,0.5,args.n_op)} #, 'C_21': np.linspace(-1,1,args.n_op)}
    #order_params = {'C_12': np.linspace(-0.5,0.5,args.n_op), 'C_21': np.linspace(-0.5,0.5,args.n_op)}
    order_params = {'C_21': np.linspace(0.2,0.3,args.n_op)}
    if args.run_stability_analysis:
        outputs, futures = run_stability_analysis(order_params, default_params, args)
        if args.save_outputs:
            fname = 'outputs_dst_'+today()+'_op_C_12_fix_C21_023.pkl'
            with open(os.path.join(proj_dir, 'postprocessing', fname), 'wb') as f:
                pickle.dump(outputs, f)
        if args.plot_bifurcation_diagrams:
            if len(order_params.keys())==1:
                plot_bifurcation_row(outputs, order_params, args)
            elif len(order_params.keys())==2:
                plot_bifurcation_grid(outputs, order_params, args)
            else:
                print("can't plot for the number of order parameters")
    
    if args.plot_phasespace_grid:
        #if ('outputs' not in globals()) & ('outputs' not in locals()):
        #    outputs = pickle.load(os.path.join(proj_dir, 'postprocessing', 'outputs_dst_20230227.pkl'))
        plot_phasespace_grid(outputs, order_params, args)
            
