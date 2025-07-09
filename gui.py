import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.widgets import TextBox, Slider, Button
import random
import traceback
from spikey.simulation import run_simulation

#graphic tool to observe parameter's affects on the network
# let's start with the parameters : v_rest, v_tresh, tau_m, tau_refrac, tau_syn_E, tau_syn_I, cm, e_rev_E, e_rev_I
# create an interactive plot, with input lines and the user can click to add spikes. 
# have sliders for each parameter
# the user will then move the slider and hit play to train the model with the new paramters
# then we create the correspondant plot. 

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.4, right=0.95, top=0.9)
ax.set_title("Spikey", fontsize=16)

current_num_inputs = 3
current_duration = 100
spike_lines = []
input_spikes = []
colors = []

def setup_plot(num_inputs, duration):
    global spike_lines, input_spikes, colors
    
    ax.clear()
    
    ax.set_xlim(0, duration)
    ax.set_ylim(0, num_inputs + 0.5)
    ax.set_xticks(np.arange(0, duration + 1, max(5, duration // 20)))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Input')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    
    ax.set_yticks(range(1, num_inputs + 1))
    labels = [f"Input {i+1}" for i in range(num_inputs)]
    ax.set_yticklabels(labels)
    
    ax.grid(True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle=':', alpha=0.5)
    
    colors = []
    for i in range(num_inputs):
        r = random.uniform(0.3, 0.9)
        g = random.uniform(0.3, 0.9)
        b = random.uniform(0.3, 0.9)
        colors.append((r, g, b))
    
    spike_lines = [[] for _ in range(num_inputs)]
    input_spikes = [[] for _ in range(num_inputs)]
    
    ax.set_title("little Spikey", fontsize=16)
    
    fig.canvas.draw_idle()

ax_inputs = plt.axes([0.25, 0.95, 0.1, 0.03])
text_box_inputs = TextBox(ax_inputs, 'Num Inputs: ', initial=str(current_num_inputs))

ax_duration = plt.axes([0.55, 0.95, 0.1, 0.03])
text_box_duration = TextBox(ax_duration, 'Duration (ms): ', initial=str(current_duration))

ax_apply = plt.axes([0.75, 0.95, 0.1, 0.03])
button_apply = Button(ax_apply, 'Apply')

def apply_settings(event):
    global current_num_inputs, current_duration
    try:
        num_inputs = int(text_box_inputs.text)
        duration = int(text_box_duration.text)
        
        if num_inputs < 1:
            print("Number of inputs must be at least 1")
            return
        if duration < 10:
            print("Duration must be at least 10ms")
            return
            
        current_num_inputs = num_inputs
        current_duration = duration
        
        setup_plot(num_inputs, duration)
        print(f"Plot updated with {num_inputs} inputs and {duration}ms duration")
    except ValueError:
        print("Please enter valid numbers")

button_apply.on_clicked(apply_settings)

def on_click(event):
    if event.inaxes != ax:
        return
        
    x = event.xdata
    y = round(event.ydata)
    
    if 0 < y <= current_num_inputs and 0 <= x <= current_duration:
        input_idx = y - 1
        input_spikes[input_idx].append(x)
        
        line = ax.vlines(x=x, ymin=y-0.2, ymax=y+0.2, color=colors[input_idx], linewidth=2)
        spike_lines[input_idx].append(line)
        
        fig.canvas.draw_idle()

setup_plot(current_num_inputs, current_duration)
fig.canvas.mpl_connect('button_press_event', on_click)

slider_left = 0.25       
slider_width = 0.65
slider_height = 0.03
slider_spacing = 0.04 
bottom_margin = 0.05

slider_positions = [bottom_margin + i*slider_spacing for i in range(7)]

ax_cm = plt.axes([slider_left, slider_positions[0], slider_width, slider_height])
ax_tau_syn_i = plt.axes([slider_left, slider_positions[1], slider_width, slider_height])
ax_tau_syn_e = plt.axes([slider_left, slider_positions[2], slider_width, slider_height])
ax_tau_refrac = plt.axes([slider_left, slider_positions[3], slider_width, slider_height])
ax_tau_m = plt.axes([slider_left, slider_positions[4], slider_width, slider_height])
ax_vthresh = plt.axes([slider_left, slider_positions[5], slider_width, slider_height])
ax_vrest = plt.axes([slider_left, slider_positions[6], slider_width, slider_height])

button_width = 0.15
button_height = 0.04
button_spacing = 0.05
button_bottom = 0.01

ax_reset_spikes = plt.axes([slider_left, button_bottom, button_width, button_height])
ax_reset_params = plt.axes([slider_left + button_width + button_spacing, 
                           button_bottom, button_width, button_height])
ax_run = plt.axes([slider_left + 2*(button_width + button_spacing),
                  button_bottom, button_width, button_height])
slider_vrest = Slider(ax_vrest, 'v_rest (mV)', -80.0, -55.0, valinit=-65.0)
slider_vthresh = Slider(ax_vthresh, 'v_thresh (mV)', -55.0, -40.0, valinit=-50.0)
slider_tau_m = Slider(ax_tau_m, 'tau_m (ms)', 5.0, 50.0, valinit=20.0)
slider_tau_refrac = Slider(ax_tau_refrac, 'tau_refrac (ms)', 0.1, 10.0, valinit=2.0)
slider_tau_syn_e = Slider(ax_tau_syn_e, 'tau_syn_E (ms)', 1.0, 10.0, valinit=5.0)
slider_tau_syn_i = Slider(ax_tau_syn_i, 'tau_syn_I (ms)', 1.0, 10.0, valinit=5.0)
slider_cm = Slider(ax_cm, 'cm (nF)', 0.1, 2.0, valinit=1.0)

button_reset_spikes = Button(ax_reset_spikes, 'Reset Spikes')
button_reset_params = Button(ax_reset_params, 'Reset Params')
button = Button(ax_run, 'Run')

def update(val):
    params = {
        'v_rest': slider_vrest.val,
        'v_thresh': slider_vthresh.val,
        'tau_m': slider_tau_m.val,
        'tau_refrac': slider_tau_refrac.val,
        'tau_syn_E': slider_tau_syn_e.val,
        'tau_syn_I': slider_tau_syn_i.val,
        'cm': slider_cm.val
    }
    print(f"Updated parameters: {params}")
    fig.canvas.draw_idle()

def reset_spikes(event):
    global spike_lines, input_spikes
    
    for input_idx in range(len(spike_lines)):
        for line in spike_lines[input_idx]:
            try:
                line.remove()
            except:
                pass
    
    spike_lines = [[] for _ in range(current_num_inputs)]
    input_spikes = [[] for _ in range(current_num_inputs)]
    
    fig.canvas.draw_idle()
    print("Spikes reset")

def reset_params(event):
    slider_vrest.reset()
    slider_vthresh.reset()
    slider_tau_m.reset()
    slider_tau_refrac.reset()
    slider_tau_syn_e.reset()
    slider_tau_syn_i.reset()
    slider_cm.reset()
    print("Parameters reset to default values.")

def run_simulation_callback(event):
    """Callback function for the 'Run' button that starts the simulation"""
    try:
        params = {
            'v_rest': slider_vrest.val,
            'v_thresh': slider_vthresh.val,
            'tau_m': slider_tau_m.val,
            'tau_refrac': slider_tau_refrac.val,
            'tau_syn_E': slider_tau_syn_e.val,
            'tau_syn_I': slider_tau_syn_i.val,
            'cm': slider_cm.val
        }
        print(f"Running simulation with parameters: {params}")
        for param, value in params.items():
            print(f" {param}: {value}")
        
        #check for input spikes
        if all(len(spikes) == 0 for spikes in input_spikes):
            print("Warning: No input spikes detected. Please add some spikes by clicking on the plot.")
            return
        
        run_simulation(params, input_spikes, colors, current_duration)
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        traceback.print_exc()

slider_vrest.on_changed(update)
slider_vthresh.on_changed(update)
slider_tau_m.on_changed(update)
slider_tau_refrac.on_changed(update)
slider_tau_syn_e.on_changed(update)
slider_tau_syn_i.on_changed(update)
slider_cm.on_changed(update)
button.on_clicked(run_simulation_callback)
button_reset_spikes.on_clicked(reset_spikes)
button_reset_params.on_clicked(reset_params)
plt.show()
