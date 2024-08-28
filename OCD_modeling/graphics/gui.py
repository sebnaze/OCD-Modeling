### Graphical User Interface
##  Author: Sebastien Naze
#   Affiliation: QIMR Berghofer 2024

# to keep clean outputs
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

# Global imports
import argparse
import dill
import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np

# Local imports
import OCD_modeling
from OCD_modeling.mcmc import get_prior, get_default_params, unpack_params
from OCD_modeling.mcmc.inference_analysis import format_param


# Defaults parameters and increment sizes for the sliders
defaults = {'C_12': 0.25, 'C_13':0.2, 'C_21':0.25, 'C_24':0.2, 'C_31':0.25, 'C_34':0., 'C_42':0.25, 'C_43':0, 
                  'G':2.5, 'eta_C_13':0.05, 'eta_C_24':0.05, 'sigma':0.075, 'sigma_C_13':0.25, 'sigma_C_24':0.25}

steps = {'C_12': 0.01, 'C_13':0.01, 'C_21':0.01, 'C_24':0.01, 'C_31':0.01, 'C_34':0.01, 'C_42':0.01, 'C_43':0.01, 
                  'G':0.05, 'eta_C_13':0.005, 'eta_C_24':0.005, 'sigma':0.005, 'sigma_C_13':0.01, 'sigma_C_24':0.01}

OCD_presets = {'C_12': 0.15, 'C_13':0.4, 'C_21':0.25, 'C_24':0.1, 'C_31':0.4, 'C_34':0., 'C_42':0.1, 'C_43':0, 
                  'G':2.25, 'eta_C_13':0.025, 'eta_C_24':0.025, 'sigma':0.07, 'sigma_C_13':0.3, 'sigma_C_24':0.3}

healthy_presets = {'C_12': 0.3, 'C_13':0., 'C_21':0.25, 'C_24':0.2, 'C_31':0.1, 'C_34':-0.4, 'C_42':0.4, 'C_43':0, 
                  'G':2.25, 'eta_C_13':0.075, 'eta_C_24':0.03, 'sigma':0.08, 'sigma_C_13':0.2, 'sigma_C_24':0.2}

# Import parameters min, max and default values from models
_, bounds = get_prior()
default_model_params, default_sim_params, default_control_params, default_bold_params = get_default_params()


# ------------------------ #
# Graphical User Interface #
# ------------------------ #
coupling_params = ['C_12', 'C_13', 'C_21', 'C_24', 'C_31', 'C_34', 'C_42', 'C_43']
global_params = ['G', 'sigma']
drift_params = ['eta_C_13', 'eta_C_24']
volatility_params = ['sigma_C_13', 'sigma_C_24']

coupling_sliders = dict()
coupling_sliders['header'] = widgets.HTML(value="<b>Coupling strength parameters</b>", layout=widgets.Layout(display='flex', justify_content='center'))
for param in coupling_params:
    coupling_sliders[param] = widgets.FloatSlider(min=bounds[param][0], max=bounds[param][1], value=defaults[param], step=steps[param], description="${}$".format(format_param(param)),
                                                 style={'handle_color':'plum'})
coupling_box = widgets.VBox(list(coupling_sliders.values()))

global_sliders = dict()
global_sliders['header'] = widgets.HTML(value="<b>Global parameters</b>", layout=widgets.Layout(display='flex', justify_content='center'))
for param in global_params:
    global_sliders[param] = widgets.FloatSlider(min=bounds[param][0], max=bounds[param][1], value=defaults[param], step=steps[param], description="${}$".format(format_param(param)),
                                               style={'handle_color':'lightblue'})
global_box = widgets.VBox(list(global_sliders.values()))

drift_sliders = dict()
drift_sliders['header'] = widgets.HTML(value="<b>Drift parameters</b>", layout=widgets.Layout(display='flex', justify_content='center'))
for param in drift_params:
    drift_sliders[param] = widgets.FloatSlider(min=bounds[param][0], max=bounds[param][1], value=defaults[param], step=steps[param], description="${}$".format(format_param(param)),
                                              style={'handle_color':'lightcoral'})
drift_box = widgets.VBox(list(drift_sliders.values()))

volatility_sliders = dict()
volatility_sliders['header'] = widgets.HTML(value="<b>Volatility parameters</b>", layout=widgets.Layout(display='flex', justify_content='center'))
for param in volatility_params:
    volatility_sliders[param] = widgets.FloatSlider(min=bounds[param][0], max=bounds[param][1], value=defaults[param], step=steps[param], description="${}$".format(format_param(param)),
                                                   style={'handle_color':'gold'})
volatility_box = widgets.VBox(list(volatility_sliders.values()))

param_box = widgets.HBox([global_box, coupling_box, drift_box, volatility_box])

presets_header = widgets.HTML(value="<b>Presets</b>", layout=widgets.Layout(display='flex', justify_content='center'))
OCD_button = widgets.Button(description='OCD', disabled=False)
healthy_button = widgets.Button(description='Healthy', disabled=False)
presets_buttons = widgets.HBox([OCD_button, healthy_button])
presets_box = widgets.VBox([presets_header, presets_buttons])

check_box = widgets.Checkbox(value=True, description='Perform stability analysis', disabled=False, indent=False)

button = widgets.Button(
    description='Run',
    disabled=False)

status = widgets.HTML(value='Ready', layout={'border': '0px'})

run = widgets.HBox([button, status])

out = widgets.Output()
gui = widgets.VBox([presets_box, param_box, check_box, run, out])
display(gui)

# ------ #
# Events #
# ------ #
@out.capture(clear_output=False, wait=True)
def run_state_space(model_params):
    """ Perform and display state space analysis (find fixed points, nullcines, compute trajectories) """ 
    args = argparse.Namespace()
    args.compute_epc = False
    args.timeout = 600
    args.n_trajs = 10
    args.save_figs=False
    args.plot_figs = True
    args.n_jobs=1
    rng = np.random.default_rng()
    warnings.filterwarnings('ignore')
    
    default_params = {'a':270, 'b': 108, 'd': 0.154, 'C_12': 0.25, 'G':2.5, 'J_N':0.2609, 'I_0':0.3, 'I_1':0.0, 'tau_S':100, 'w':0.9, 'gam':0.000641}
    
    plt.figure(figsize=[12,3])
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,3)
    
    # Ventromedial circuit
    order_params = {'C_12': np.array([model_params['C'][0,2]]), 'C_21': np.array([model_params['C'][2,0]])}
    outputs, _ = OCD_modeling.analysis.run_stability_analysis(order_params, default_params, args)

    out = dill.loads(outputs[0])
    output = out['output']    
    OCD_modeling.analysis.plot_phasespace(output['model'], output['fps'], output['ncs'], output['trajs'], ax=ax1, args=args)
    plt.getp(ax1, 'children')[0].set(alpha=1)
    plt.xlabel('OFC', fontsize=13)
    plt.ylabel('NAcc', fontsize=13)
    plt.title("Ventromedial circuit (affective)", fontdict={'fontsize': 13} )

    
    # Dorsolateral circuit
    order_params = {'C_12': np.array([model_params['C'][1,3]]), 'C_21': np.array([model_params['C'][3,1]])}
    outputs, _ = OCD_modeling.analysis.run_stability_analysis(order_params, default_params, args)

    out = dill.loads(outputs[0])
    output = out['output']    
    OCD_modeling.analysis.plot_phasespace(output['model'], output['fps'], output['ncs'], output['trajs'], ax=ax2, args=args)
    plt.getp(ax2, 'children')[0].set(alpha=1)
    plt.xlabel('PFC', fontsize=13)
    plt.ylabel('Putamen', fontsize=13)
    plt.title("Dorsolateral circuit (cognitive)", fontdict={'fontsize': 13} )


@out.capture(clear_output=True, wait=True)
def run_model(button):
    """ Simulate time series of model with selected parameters, derive and display BOLD signal and functional connectivity """ 
    status.value='Running... please wait.'
    sliders = {**global_sliders, **coupling_sliders, **drift_sliders, **volatility_sliders}
    params = dict()
    for param,slider in sliders.items():
        if param!='header': params[param] = slider.value 

    model_params, sim_params, control_params, bold_params = unpack_params(params)
    
    # state space analysis
    if check_box.value:
        run_state_space(model_params)
    
    # BOLD time series and FC
    model = OCD_modeling.models.ReducedWongWangOU(**model_params)
    model.run(**sim_params)
    OCD_modeling.models.compute_bold(model, t_range=[1200, 5400])
    OCD_modeling.models.plot_bold(model, labels=['OFC', 'PFC', 'NAcc', 'Put'])

    status.value='Ready'


# Presets
def presets_params(presets):
    """ Set slider values to input presets """
    sliders = {**global_sliders, **coupling_sliders, **drift_sliders, **volatility_sliders}
    for param,slider in sliders.items():
        if param in presets.keys():
            slider.value = presets[param]

def preset_OCD(OCD_button):
    """ Reset model parameters to OCD values """ 
    presets_params(OCD_presets)
    
def preset_healthy(healthy_button):
    """ Reset model parameters to OCD values """ 
    presets_params(healthy_presets)

# Button handlers    
OCD_button.on_click(preset_OCD)
healthy_button.on_click(preset_healthy)
button.on_click(run_model)

