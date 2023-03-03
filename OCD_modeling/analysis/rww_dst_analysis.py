####    OCD Modeling: Reduced Wong-Wang Model Analysis using Dynamical Systems Toolbox
###      
##      Author: Sebastien Naze
#       QIMR 2023

from abc import ABC
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
import copy
import datetime
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
import pickle
import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp
import scipy
import sympy as sp
import time

from OCD_modeling.utils import utils
from OCD_modeling.models import ReducedWongWang as RWW


proj_dir = os.path.join(utils.get_working_dir(), 'lab_lucac/sebastiN/projects/OCD_modeling')

rng = np.random.default_rng()

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
    nulls_x, nulls_y = pp.find_nullclines(model, 'S1', 'S2', n=6, eps=1e-8, max_step=0.01, fps=fp_coords)
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



def compute_equilibrium_point_curve(cont, fps, free_params):
    """ find equilibrium point curve(s) of the system, starting from each fixed point (if exist)""" 
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



def stability_analysis(order_params, default_params, out_queue, args, pdomain={'C_21':[-1, 1]}):
    """ Create model and analyse dynamics using PyDSTool """
    for k,v in order_params.items(): 
        default_params[k] = v
    # fixed point and nullclines
    model = create_model(default_params, args)
    model, fps, nullclines = get_fixed_points(model, default_params)
    points = compute_tajectories(model, args.n_trajs)
    # EP-C
    model.set(pdomain=pdomain)
    cont = dst.ContClass(model)
    try:
        compute_equilibrium_point_curve(cont, fps, list(pdomain.keys()))
    except:
        return copy.deepcopy(order_params)
    out = {'dilled_cont':dill.dumps(cont, byref=True), 'fps':fps, 'ncs':nullclines}
    output = copy.deepcopy(order_params)
    output['output'] = out
    out_queue.put(copy.deepcopy(output))
    #return output



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
    """ Starts a pool of parallel processes """
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
    #print(futures)
    outputs = []
    while not out_queue.empty():
        outputs.append(out_queue.get_nowait())
    return outputs, futures
    

def plot_phasespace(model, fps, nullclines, points, n=5, eps=1e-8, ax=None, args=None):
    """ Plot vector field, nullclines and fixed points of a model previously set with parameters """
    plt.gca(ax)
    pp.plot_PP_vf(model, 'S1', 'S2', scale_exp=-1)
    pp.plot_PP_fps(fps, do_evecs=True, markersize=5)
    plt.plot(nullclines[0][:,0], nullclines[0][:,1], 'b')
    plt.plot(nullclines[1][:,0], nullclines[1][:,1], 'g')
    for pts in points:
        plt.plot(pts['S1'], pts['S2'], linewidth=1)
    plt.show(block=False)


def get_grid_inds(output, order_params):
    """ get the indices (i,j) of the order parameters from output """
    o_pars = np.sort([k for k in output.keys() if k!='output'])
    i = [x for x,val in enumerate(order_params[o_pars[0]]) if val==output[o_pars[0]]]
    j = [y for y,val in enumerate(order_params[o_pars[1]]) if val==output[o_pars[1]]]
    return o_pars, i[0],j[0]

def plot_phasespace_grid(outputs, order_params, args=None):
    """ Plot a grid of graphs from outputs """
    fig = plt.figure(figsize=[16,16])
    p1s, p2s = list(order_params.values())[0], list(order_params.values())[1]
    gs = plt.GridSpec(nrows=len(p1s), ncols=len(p2s))
    
    for output in outputs:
        o_pars,i,j = get_grid_inds(output, order_params)
        ax = fig.add_subplot(gs[i,j])
        plot_phasespace(output['cont'].model, output['fps'], output['ncs'], output['pts'], ax=ax, args=args)

        plt.axis('tight')
        plt.title("{}={:.3f}  {}={:.3f}".format(o_pars[0], p1s[i], o_pars[1], p2s[j]))
        plt.xlabel('S1')
        plt.ylabel('S2')
    

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
                cont.display(axes=ax, coords=['C_21', 'S2'], stability=True, color='blue')
                cont.display(axes=ax, coords=['C_21', 'S1'], stability=True, color='orange')
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
    plt.rcParams.update({'font.size':6, 'axes.titlesize':'medium'})
    fig = plt.figure(figsize=[20,5])
    p1s = list(order_params.values())[0]
    gs = plt.GridSpec(nrows=1, ncols=len(p1s))
    for output in outputs: 
        o_par = list(order_params.keys())[0]
        i = [x for x,val in enumerate(order_params[o_par]) if val==output[o_par]][0]
        ax = fig.add_subplot(gs[0,i])
        if 'output' in output.keys():
            cont = dill.loads(output['output']['dilled_cont'])
            try:
                cont.display(axes=ax, coords=['C_21', 'S2'], stability=True, color='blue')
                cont.display(axes=ax, coords=['C_21', 'S1'], stability=True, color='orange')
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
            plt.xlim([-1,1])
            plt.ylim([0,1])
        else:
            plt.xlabel("")
            plt.xticks([])
            plt.ylabel("")
            plt.yticks([])
            plt.title("{}={:.3f}".format(o_par, p1s[i]))
    plt.show(block=False)



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
    parser.add_argument('--n_jobs', type=int, default=12, action='store', help='number of processes used in parallelization')
    parser.add_argument('--n_trajs', type=int, default=0, action='store', help='number of trajectories (traces) to compute for phase space projection')
    parser.add_argument('--n_op', type=int, default=10, action='store', help='number of values taken for each order parameters')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    default_params = {'a':270, 'b': 108, 'd': 0.154, 'C_21': 0, 'G':2.5, 'J_N':0.2609, 'I_0':0.3, 'I_1':0.0, 'tau_S':100, 'w':0.9, 'gam':0.000641}
    order_params = {'C_12': np.linspace(-1,1,args.n_op), 'I_0': np.linspace(0.2,0.5,args.n_op)} #, 'C_21': np.linspace(-1,1,args.n_op)}
    if args.run_stability_analysis:
        outputs, futures = run_stability_analysis(order_params, default_params, args)
        if args.save_outputs:
            today = datetime.datetime.now().strftime("%Y%m%d")
            fname = 'outputs_dst_'+today+'_C_12_I_0.pkl'
            with open(os.path.join(proj_dir, 'postprocessing', fname), 'wb') as f:
                pickle.dump(outputs, f)
        if args.plot_figs:
            #plot_phasespace_grid(outputs, order_params)
            if len(order_params.keys())==1:
                plot_bifurcation_row(outputs, order_params, args)
            elif len(order_params.keys())==2:
                plot_bifurcation_grid(outputs, order_params, args)
            else:
                print("can't plot for the number of order parameters")
            
