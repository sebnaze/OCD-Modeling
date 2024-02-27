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
    """ Create the Dynamical System in PyDSTool 
    
        .. math:: \dot{S_i} = - \cfrac{S_i}{\\tau_S} + (1 - S_i) \gamma H(x_i) + \sigma v_i
    
    with

        .. math:: H(x) = \cfrac{ax-b}{1-\exp{(-d(ax-b))}}

    and

        .. math:: x_i = w J_N S_i + G J_N \sum_j{C_{ij} S_j} + I_0 

        
    Parameters
    ----------
        params: dict
            Parameters of the model
        args: Argparse.Namespace
            Optional arguments 

    Returns
    -------
        rww: PyDSTool.Vode_ODEsystem
            PyDSTool object containing the dynamical system 
            
    """
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
    """ Get model's nullclines :math:`\\frac{dS_1}{dt}=0` and :math:`\\frac{dS_2}{dt}=0` 
    and fixed points :math:`\\frac{dS_1}{dt}=\\frac{dS_2}{dt}=0` for a given set of parameters.
    
    Parameters
    ----------
        model: PyDSTool.Vode_ODEsystem
            Model object in PyDSTool.
        params: dict
            Model parameters.
        xdomain: dict
            Variable and lower/upper bounds.
        args: Argparse.Namespace
            Optional extra arguments.

    Returns
    -------
        model: PyDSTool.Vode_ODEsystem
            Updated model object with given parameters.
        fps: PyDSTool.Toolbox.phaseplane.fixedpoint_2D 
            Fixed points of the system.
        (nulls_x, nulls_y): tuple 
            Tuple containing nullclines (arrays of paired xs and ys).
    """
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


def compute_trajectories(model, n, tdata=[0,1000]):
    """ Compute n trajectories from model, each with different initial conditions.
    
    Parameters
    ----------
        model: PyDSTool.Vode_ODEsystem
            PyDSTool model object.
        n: int
            Number of trajectories to compute.
        tdata: list
            Time interval of the saved trajectories.
    """
    points = []
    for i in range(n):
        model.set(ics={'S1':rng.uniform(), 'S2':rng.uniform()},
                  tdata=tdata)
        traj = model.compute(f"traj {i}")
        pts = traj.sample()
        points.append(pts)
    return points



def compute_equilibrium_point_curve(model, fps, pdomain):
    """ Find equilibrium point curve(s) of the system, starting from each fixed point (if exist).

    Parameters
    ----------
        model: PyDSTool.Vode_ODEsystem
            Model object in PyDSTool.
        fps: PyDSTool.Toolbox.phaseplane.fixedpoint_2D
            Fixed points of the system.
        pdomain: dict
            Free variable (or order parameter) to perform the bifurcation analyis from.

    Returns
    -------
        cont: PyDSTool.ContClass
            PyDSTool Continuation Class object populated with equilibrium point curves.
    """ 
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



def stability_analysis(order_params, default_params, out_queue, args, pdomain={'C_12':[-1, 2]}):
    """ Create model and analyse dynamics using PyDSTool.
    
    Parameters
    ----------
        order_params: dict
            Fixed parameters, for which to analyse the system using discretized values, for example ``{'C_21': np.linspace(0.2,0.8,4)}``.
        default_params: dict
            Default model's parameters.
        out_queue: Queue
            Queue to put results in (used for parallel computation) for each values of order parameter.
        args: Argparse.Namespace 
            Structure of necessary options. For example, must include ``args.compute_epc = True`` to compute equilibrium point curves.
        pdomain: dict
            Free variable (or order parameter) to perform the bifurcation analyis from.

    Returns
    -------
        None
            A dict with model, nullclines (ncs), fixed points (fps), trajectories (trajs), 
            and a pickled (dilled) continuation object (if equilibrium curves are asked in args), is appended to the queue.
    
    """
    for k,v in order_params.items(): 
        default_params[k] = v
    
    out = dict()
    
    # fixed point and nullclines
    model = create_model(default_params, args)
    model, fps, nullclines = get_fixed_points(model, default_params)
    points = compute_trajectories(model, args.n_trajs)
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
        creating a child process with a set timeout per child process, such that the stbaility analysis
        does not hang waiting for the continuation to terminate if it does not converge. 
        
    Parameters
    ----------
        order_params: dict 
            Fixed parameters, for which to analyse the system using discretized values, for example ``{'C_21': np.linspace(0.2,0.8,4)}``.
        default_params: dict
            Default model's parameters. 
        out_queue: Queue
            Output queue on which to append the results.
        args: Argparse.Namespace
            Structure of necessary options. For example, must include ``args.compute_epc = True`` to compute equilibrium point curves.
    """
    proc = multiprocessing.Process(target=stability_analysis, args=(order_params, default_params, out_queue, args))
    proc.start()
    # wait for the process until timeout
    proc.join(args.timeout)
    # if process is still running after timeout, force terminating it 
    if proc.is_alive():
        print(f"Stability analysis for {order_params} took too long, aborted after {args.timeout}s")
        proc.terminate()



def run_stability_analysis(order_params, default_params, args):
    """ Starts a pool of parallel processes to run the stability analysis.

    Parameters
    ----------
        order_params: dict 
            Fixed parameters, for which to analyse the system using discretized values, for example ``{'C_21': np.linspace(0.2,0.8,4)}``.
        default_params: dict
            Default model's parameters. 
        args: Argparse.Namespace
            Structure of necessary options. For example, must include ``args.compute_epc = True`` to compute equilibrium point curves.

    Returns
    -------
        outputs: list of dict
            Stability analysis of each values (or combination of values) given in order_params.
        futures: list of concurrent.futures.Future
            (deprecated) if using futures.concurrent library for parallel process, return the list of Future objects (https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future).
    """
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
    

def plot_phasespace(model, fps, nullclines, trajs, ax=None, args=None):
    """ Plot vector field, nullclines and fixed points of a model previously set with parameters.
    
    Parameters
    ----------
        model: PyDSTool.Vode_ODEsystem 
            PyDSTool model object.
        fps: PyDSTool.Toolbox.phaseplane.fixedpoint_2D
            Fixed points of he system
        nullclines: list
            list of PyDSTool nullcline objects, containing arrays of xs and ys of nullclines.
        trajs: list of dict
            Simulated trajectories.
        ax: matplotlib.Axis 
            Axis in which to plot the phasespace.
    """
    plt.sca(ax)
    pp.plot_PP_vf(model, 'S1', 'S2', scale_exp=-2)
    pp.plot_PP_fps(fps, do_evecs=True, markersize=6)
    plt.plot(nullclines[0][:,0], nullclines[0][:,1], 'b', lw=3)
    plt.plot(nullclines[1][:,0], nullclines[1][:,1], 'g', lw=3)
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
    """ Plot a grid of phasespaces from stability analysis outputs.
    
    Parameters
    ----------
        outputs: list 
            Outputs from stability analysis.
        order_params: dict 
            Fixed parameters, for which to analyse the system using discretized values, for example ``{'C_21': np.linspace(0.2,0.8,4)}``.
        args: Argparse.Namespace
            Optional extra arguments in argprase Namespace, such as ``args.save_figs=True`` to save figure and ``args.plot_figs=True`` to plot figures.
      
    """
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
        ttl = "$C_{12}$=%s        $C_{21}$=%s" %("{:.2f}".format(p1s[i]),"{:.2f}".format(p2s[j]))
        plt.title(ttl, fontdict={'fontsize': 13} )#.format(p1s[i], p2s[j]))
        
        if i==len(p1s)-1:
            plt.xlabel('$S_1$', fontsize=13)
        if j==0:
            plt.ylabel('$S_2$', fontsize=13)
        plt.tight_layout()

        # makes quiver a bit more transparent and arrows a bit larger
        plt.getp(ax, 'children')[0].set(alpha=0.7)
        #for path in plt.getp(plt.getp(ax,'children')[0], 'paths'):
        #    path.vertices = path.vertices*10

    #pdb.set_trace()
    if args.save_figs:
        plt.savefig(os.path.join(proj_dir, 'img', 'phase_space'+today()+'.svg'), format='svg', transparent=True)
    if args.plot_figs:
        plt.show()
    

def plot_phasespace_row(outputs, order_params, rww=None, t_range=None, args=None):
    """ Plot a row of graphs from outputs """
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.size':11, 'axes.titlesize':'medium', 'mathtext.default': 'regular'})
    fig = plt.figure(figsize=[12,3])
    p1s = list(order_params.values())[0]
    gs = plt.GridSpec(nrows=1, ncols=len(p1s))
    #o_pars = np.sort([k for k in output.keys() if k!='output'])
    #i = [x for x,val in enumerate(order_params[o_pars[0]]) if val==output[o_pars[0]]]
    for i,res in enumerate(outputs):
        out = dill.loads(res)
        output = out['output']    
        
        ax = fig.add_subplot(gs[0,i])
        plot_phasespace(output['model'], output['fps'], output['ncs'], output['trajs'], ax=ax, args=args)

        plt.axis('tight')
        #plt.title("{}={:.3f}  {}={:.3f}".format(o_pars[0], p1s[i], o_pars[1], p2s[j]))
        ttl = "$C_{2 \leftarrow 1}$=%s" %("{:.3f}".format(p1s[i]))
        plt.title(ttl, fontdict={'fontsize': 11} )#.format(p1s[i], p2s[j]))
        plt.xlabel('$S_1$')
        if i==0:
            plt.ylabel('$S_2$')
        plt.tight_layout()

        # makes quiver a bit more transparent and arrows a bit larger
        plt.getp(ax, 'children')[0].set(alpha=0.7)
        #for path in plt.getp(plt.getp(ax,'children')[0], 'paths'):
        #    path.vertices = path.vertices*10

        if rww!=None:
            if t_range!=None:
                start = t_range[0]*rww.sf
                stop = t_range[1]*rww.sf
            else:
                start = 0
                stop = rww.S_rec.shape[0]
            plt.plot(rww.rec_C_12[start:stop], rww.S_rec[start:stop,0], lw=0.5, color='blue', alpha=0.4)
            plt.plot(rww.rec_C_12[start:stop], rww.S_rec[start:stop,1], lw=0.5, color='red', alpha=0.4)

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

def plot_bifurcation_row(outputs, order_params, rww=None, t_range=None, args=None):
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
        if rww!=None:
            if t_range!=None:
                start = t_range[0]*rww.sf
                stop = t_range[1].sf
            else:
                start = 0
                stop = rww.S_rec.shape[0]
            plt.plot(rww.rec_C_12[start:stop], rww.S_rec[start:stop,0], lw=0.5, color='blue', alpha=0.4)
            plt.plot(rww.rec_C_12[start:stop], rww.S_rec[start:stop,1], lw=0.5, color='red', alpha=0.4)

    plt.tight_layout()
    
    if args.save_figs:
        fname = os.path.join(proj_dir, 'img', 'bifurcation_diagram_023_'+today()+'.svg')
        plt.savefig(fname)
    
    if args.plot_figs:
        plt.show(block=False)
    else:
        fig.close()


def plot_timeseries_phasespace_bif(outputs, rww, args):
    """ Show S1, S2, C_12 timeseries, S1-S2 phase space with trajectories and C_12 S1/S2 phase space 
    with bifurcation diagram.
    
    Parameters
    ----------
        outputs: list 
            Outputs from stability analysis.
        rww: OCD_modeling.models.ReducedWongWangOU
            Model instance that ran. 
        args: Argparse.Namespace
            Extra options.        
    """
    t_range = [2800,6000]

    ticks = [0 ,0.3, 0.6, 0.9]

    start = t_range[0]*rww.sf
    stop = t_range[1]*rww.sf


    fig = plt.figure(figsize=[9,6])
    gs = plt.GridSpec(nrows=2, ncols=3)
    sub_gs = gs[1,0:3].subgridspec(3,1)

    # Time series
    #------------
    ax = fig.add_subplot(sub_gs[0,0])
    ax.plot(rww.t[start:stop], rww.S_rec[start:stop,0], color='dodgerblue')
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.right.set_visible(False)
    plt.xticks([], label='')
    plt.yticks([0,1])
    plt.ylabel('$S_1$', rotation=0, labelpad=15, fontsize=12)

    ax = fig.add_subplot(sub_gs[1,0])
    ax.plot(rww.t[start:stop], rww.S_rec[start:stop,1], color='forestgreen')
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.right.set_visible(False)
    plt.xticks([], label='')
    plt.yticks([0,1])
    plt.ylabel('$S_2$', rotation=0, labelpad=15, fontsize=12)

    ax = fig.add_subplot(sub_gs[2,0])
    ax.hlines(y=0, xmin=t_range[0], xmax=t_range[1], lw=0.5, linestyle='--', color='gray')
    ax.plot(rww.t[start:stop], rww.rec_C_12[start:stop], lw=0.25, color='orange')
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    #plt.yticks([-0.5,0.75])
    plt.yticks([-1.2,1.6])
    plt.ylabel('$C_{12}$', rotation=0, labelpad=0, fontsize=12)

    xticks = np.arange(t_range[0], t_range[1], 600) # 10min ticks
    plt.xticks(xticks, labels=np.array((xticks-t_range[0])/60, dtype=int))
    ax.set_xlabel('time (min)', fontsize=12)


    # S1-S2 State space
    #------------------
    output = dill.loads(outputs[3])['output']
    ax = fig.add_subplot(gs[0,1])
    ax.plot(output['ncs'][1][:,0], output['ncs'][1][:,1], color='forestgreen', lw=3)
    ax.plot(output['ncs'][0][:,0], output['ncs'][0][:,1], color='dodgerblue', lw=3)
    ax.plot(rww.S_rec[start:stop,0], rww.S_rec[start:stop,1], lw=1, color='gray', alpha=1)
    for fp in output['fps']:
        x,y = fp.toarray()
        ax.scatter(x, y, color='black', s=80)
        
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.set_xlabel('$S_1$', fontsize=12)
    ax.set_ylabel('$S_2$', rotation=0, labelpad=15, fontsize=12)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlim([-0.05,1])
    ax.set_ylim([-0.05,1])
    plt.tight_layout()


    # S - C_12 state space
    ax = fig.add_subplot(gs[0,2])
    cont = dill.loads(output['dilled_cont'])
    cont.display(axes=ax, coords=['C_12', 'S1'], stability=True, color='blue', linewidth=1.6)
    cont.display(axes=ax, coords=['C_12', 'S2'], stability=True, color='red', linewidth=1.6)

    plt.plot(rww.rec_C_12[start:stop], rww.S_rec[start:stop,0], lw=0.25, color='blue', alpha=0.2, label='$S_1$')
    plt.plot(rww.rec_C_12[start:stop], rww.S_rec[start:stop,1], lw=0.25, color='red', alpha=0.2, label='$S_2$')
    ax.spines.top.set_visible(False)
    ax.spines.left.set_color('blue')
    ax.spines.right.set_color('red')
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_ylim([-0.05,1])
    ax.set_xlim([-0.6,1.6])
    ax.tick_params(axis='y', which='both', labelleft='on', left=True)
    ax.tick_params(axis='y', which='both', labelright='on', right=True)
    plt.xlabel('$C_{12}$', fontsize=12)
    plt.ylabel('$S_1$', rotation=0, fontsize=12)
    plt.title('')

    plt.tight_layout()

    if args.save_figs:
        fname = os.path.join(proj_dir, 'img', 'single_pathway_model'+today()+'.svg')
        plt.savefig(fname)



def get_parser():
    """ parsing global script argument """
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_model', default=False, action='store_true', help='create symbolic model')
    parser.add_argument('--compute_nullclines', default=False, action='store_true', help='compute nullclines numerically')
    parser.add_argument('--compute_epc', default=False, action='store_true', help='compute equilibrium point curves numerically')
    parser.add_argument('--save_model', default=False, action='store_true', help='save symbolic model with its computed attributes')
    parser.add_argument('--run_stability_analysis', default=False, action='store_true', help='run stability analysis: find fixed points (semi-analytically) and perform linear stability analysis around them')
    parser.add_argument('--load_stability_analysis', default=False, action='store_true', help='load previously completed stability analysis')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--plot_phasespace_grid', default=False, action='store_true', help='plot grid of phase spaces using discretized order parameters')
    parser.add_argument('--plot_bifurcation_diagrams', default=False, action='store_true', help='plot grid of bifurcation diagrams using discretized order parameters')
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save analysis outputs')
    parser.add_argument('--timeout', type=int, default=3000, action='store', help='timeout of the stability analysis (per parameter combination invoked)')
    parser.add_argument('--n_jobs', type=int, default=20, action='store', help='number of processes used in parallelization')
    parser.add_argument('--n_trajs', type=int, default=10, action='store', help='number of trajectories (traces) to compute for phase space projections')
    parser.add_argument('--n_op', type=int, default=5, action='store', help='number of values taken for each order parameters')
    parser.add_argument('--load_sample_rww', default=False, action='store_true', help='load a sample of ReducedWongWang model object previously ran for illustration')
    parser.add_argument('--plot_timeseries_phasespace_bif', default=False, action='store_true', help='plot neat figure of timeseries, phase space and bifurcations with trajectories (paper quality)')
    return parser


if __name__=='__main__':
    args = get_parser().parse_args()
    default_params = {'a':270, 'b': 108, 'd': 0.154, 'C_12': 0.25, 'G':2.5, 'J_N':0.2609, 'I_0':0.3, 'I_1':0.0, 'tau_S':100, 'w':0.9, 'gam':0.000641}
    #order_params = {'C_12': np.linspace(-1,1,args.n_op), 'I_0': np.linspace(0.2,0.5,args.n_op)} #, 'C_21': np.linspace(-1,1,args.n_op)}
    order_params = {'C_12': np.linspace(-0.5,0.5,args.n_op), 'C_21': np.linspace(-0.5,0.5,args.n_op)}
    #order_params = {'C_21': np.linspace(0.2,0.8,args.n_op)}
    
    if args.load_stability_analysis:
        #fname = os.path.join(proj_dir, 'postprocessing', 'outputs_dst__20230925_op_C_12_fix025_C21_var023.pkl')
        fname = os.path.join(proj_dir, 'postprocessing', 'outputs_dst__20230925_op_C_12_fix025_C21_var0_1_10.pkl')
        with open(fname, 'rb') as f:
            outputs = pickle.load(f) 
    
    elif args.run_stability_analysis:
        outputs, futures = run_stability_analysis(order_params, default_params, args)
        if args.save_outputs:
            fname = 'outputs_dst_'+today()+'_phasespace_grid.pkl'
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
        if len(order_params.keys())==1:
            plot_phasespace_row(outputs, order_params, args)
        elif len(order_params.keys())==2:
            plot_phasespace_grid(outputs, order_params, args)

    if args.load_sample_rww:
        with open(os.path.join(proj_dir, 'postprocessing', 'sample_rww.pkl'), 'rb') as f:
            rww = pickle.load(f)
            
    if args.plot_timeseries_phasespace_bif:
        plot_timeseries_phasespace_bif(outputs, rww, args)