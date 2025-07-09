import pyNN.brian2 as sim
import matplotlib.pyplot as plt
import numpy as np
import logging

class NeuralSimulation:
    def __init__(self):
        self.sim = sim
        self.excitatory_count = 5
        self.inhibitory_count = 2
        self.sim_duration = 0
        self.populations = {}
        self.projections = {}
        self.spike_sources = []
        self.input_to_excitatory_projections = []

    def setup(self, params, input_spikes, duration):
        """
        Set up the neural simulation
        
        Parameters:
        params (dict): Neuron parameters from the GUI sliders
        input_spikes (list): List of spike times for each input
        duration (float): Duration of the simulation in ms
        """
        self.sim_duration = duration + 50  #just a bit longer than the input duration
        
        #convert spike times to arrays
        spike_arrays = []
        for i, spikes in enumerate(input_spikes):
            if spikes:  #check for spikes
                spike_arrays.append(np.array(spikes))
            else:
                spike_arrays.append(np.array([]))
        
        if 'brian2' in self.sim.__name__.lower():
            self.sim.setup(timestep=0.1, minimal_delay=0.1)
        else:
            self.sim.setup(timestep=0.1)
        
        cell_params = {
            'v_rest': params['v_rest'],      # Resting membrane potential (mV)
            'v_thresh': params['v_thresh'],  # Spike threshold (mV)
            'tau_m': params['tau_m'],        # Membrane time constant (ms)
            'tau_refrac': params['tau_refrac'],  # Refractory period (ms)
            'tau_syn_E': params['tau_syn_E'],  # Excitatory synapse time constant (ms)
            'tau_syn_I': params['tau_syn_I'],  # Inhibitory synapse time constant (ms)
            'cm': params['cm'],              # Membrane capacitance (nF)
            'i_offset': 0.0,                 # Offset current (nA)
            'v_reset': params['v_rest'] + 5.0  # Reset potential after spike (mV)
        }
        
        #create input spike sources
        self.spike_sources = []
        for i, spike_array in enumerate(spike_arrays):
            if len(spike_array) > 0:
                spike_source = self.sim.SpikeSourceArray(spike_times=spike_array.tolist())
                self.spike_sources.append(self.sim.Population(1, spike_source, label=f"input_{i}"))
            else:
                #if there's no spikes for this input still create a source
                spike_source = self.sim.SpikeSourceArray(spike_times=[])
                self.spike_sources.append(self.sim.Population(1, spike_source, label=f"input_{i}"))
        
        if hasattr(self.sim, 'IF_cond_exp'):
            neuron_model = self.sim.IF_cond_exp
        else:
            #if the conductance-based model isn't available
            neuron_model = self.sim.IF_curr_exp
            
        self.populations['excitatory'] = self.sim.Population(
            self.excitatory_count, 
            neuron_model(**cell_params),
            label="Excitatory"
        )
        
        self.populations['inhibitory'] = self.sim.Population(
            self.inhibitory_count,
            neuron_model(**cell_params),
            label="Inhibitory"
        )
        
        self.populations['excitatory'].initialize(v=params['v_rest'] + np.random.uniform(0, 10, size=self.excitatory_count))
        self.populations['inhibitory'].initialize(v=params['v_rest'] + np.random.uniform(0, 10, size=self.inhibitory_count))
        
        stdp_model = self._create_stdp_model()
        
        self.input_to_excitatory_projections = []
        for i, source in enumerate(self.spike_sources):
            if stdp_model:
                projection = self.sim.Projection(
                    source, self.populations['excitatory'],
                    connector=self.sim.AllToAllConnector(),
                    synapse_type=stdp_model,
                    receptor_type="excitatory",
                    label=f"Input{i} to Excitatory"
                )
            else:
                #static synapse if STDP isn't available
                projection = self.sim.Projection(
                    source, self.populations['excitatory'],
                    connector=self.sim.AllToAllConnector(),
                    synapse_type=self.sim.StaticSynapse(weight=0.2, delay=1.0),
                    receptor_type="excitatory",
                    label=f"Input{i} to Excitatory"
                )
            self.input_to_excitatory_projections.append(projection)
        
        #connect excitatory to inhibitory
        self.projections['e_to_i'] = self.sim.Projection(
            self.populations['excitatory'], self.populations['inhibitory'],
            connector=self.sim.AllToAllConnector(),
            synapse_type=self.sim.StaticSynapse(weight=0.3, delay=1.0),
            receptor_type="excitatory",
            label="E to I"
        )
        
        #connect inhibitory to excitatory (feedback inhibition)
        self.projections['i_to_e'] = self.sim.Projection(
            self.populations['inhibitory'], self.populations['excitatory'],
            connector=self.sim.AllToAllConnector(),
            synapse_type=self.sim.StaticSynapse(weight=0.2, delay=1.0),
            receptor_type="inhibitory",
            label="I to E"
        )
        
        self.populations['excitatory'].record(['spikes', 'v'])
        self.populations['inhibitory'].record(['spikes', 'v'])

    def _create_stdp_model(self):
        """Create STDP model based on the backend being used"""
        tau_plus = 20.0  # ms
        tau_minus = 20.0  # ms
        w_max = 0.5      # maximum weight
        A_plus = 0.01    # amplitude of potentiation
        A_minus = 0.0105 # amplitude of depression
        
        try:
            backend_name = self.sim.__name__
            
            if 'brian2' in backend_name.lower():
                print("Using Brian2 backend STDP implementation")
                print("Brian2 backend detected - using static synapses instead of STDP")
                return None
            elif 'nest' in backend_name.lower():
                print("Using NEST backend STDP implementation")
                #NEST backend has A_plus and A_minus
                return self.sim.STDPMechanism(
                    timing_dependence=self.sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus),
                    weight_dependence=self.sim.AdditiveWeightDependence(w_min=0, w_max=w_max, A_plus=A_plus, A_minus=A_minus),
                    weight=0.2, delay=1.0
                )
            else:
                print(f"Using generic STDP implementation for {backend_name} backend")
                # Try simpler constructor without the A_plus/A_minus parameters
                return self.sim.STDPMechanism(
                    timing_dependence=self.sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus),
                    weight_dependence=self.sim.AdditiveWeightDependence(w_min=0, w_max=w_max),
                    weight=0.2, delay=1.0
                )
        except Exception as e:
            print(f"Warning: STDP setup failed: {str(e)}")
            print("Using static synapses instead")
            return None
    
    def run(self):
        """Run the simulation for the specified duration"""
        print(f"Running simulation for {self.sim_duration}ms...")
        self.sim.run(self.sim_duration)
        
    def get_data(self):
        """Retrieve simulation results"""
        print("Retrieving simulation results...")
        excitatory_data = self.populations['excitatory'].get_data()
        inhibitory_data = self.populations['inhibitory'].get_data()
        return excitatory_data, inhibitory_data
    
    def plot_results(self, params, input_spikes, colors):
        """Plot the simulation results"""
        excitatory_data, inhibitory_data = self.get_data()
        
        result_fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 2])
        
        ax_inputs = plt.subplot(gs[0])
        for i, spikes in enumerate(input_spikes):
            if spikes:
                ax_inputs.scatter(spikes, [i+1] * len(spikes), marker='|', s=100, color=colors[i], label=f"Input {i+1}")
        ax_inputs.set_title("Input Spike Trains")
        ax_inputs.set_xlabel("Time (ms)")
        ax_inputs.set_ylabel("Input")
        ax_inputs.set_yticks(range(1, len(input_spikes)+1))
        ax_inputs.set_xlim(0, self.sim_duration)
        ax_inputs.grid(True)
        
        # Plot excitatory neuron spikes
        ax_exc = plt.subplot(gs[1])
        for segment in excitatory_data.segments:
            if segment.spiketrains:
                for i, spiketrain in enumerate(segment.spiketrains):
                    if len(spiketrain) > 0:
                        spike_times = spiketrain.times.magnitude
                        ax_exc.scatter(spike_times, [i+1] * len(spike_times), marker='|', s=100, color='r')
        ax_exc.set_title("Excitatory Neuron Spikes")
        ax_exc.set_xlabel("Time (ms)")
        ax_exc.set_ylabel("Neuron Index")
        ax_exc.set_yticks(range(1, self.excitatory_count+1))
        ax_exc.set_xlim(0, self.sim_duration)
        ax_exc.grid(True)
        
        ax_inh = plt.subplot(gs[2])
        for segment in inhibitory_data.segments:
            if segment.spiketrains:
                for i, spiketrain in enumerate(segment.spiketrains):
                    if len(spiketrain) > 0:
                        spike_times = spiketrain.times.magnitude
                        ax_inh.scatter(spike_times, [i+1] * len(spike_times), marker='|', s=100, color='b')
        ax_inh.set_title("Inhibitory Neuron Spikes")
        ax_inh.set_xlabel("Time (ms)")
        ax_inh.set_ylabel("Neuron Index")
        ax_inh.set_yticks(range(1, self.inhibitory_count+1))
        ax_inh.set_xlim(0, self.sim_duration)
        ax_inh.grid(True)
        
        #membrane potentials
        ax_v = plt.subplot(gs[3])
        plot_colors = plt.cm.viridis(np.linspace(0, 1, self.excitatory_count))
        
        #membrane potential of excitatory neurons
        for segment in excitatory_data.segments:
            if hasattr(segment, 'analogsignals') and len(segment.analogsignals) > 0:
                vm = segment.analogsignals[0]
                for i in range(self.excitatory_count):
                    ax_v.plot(vm.times.magnitude, vm[:, i].magnitude, 
                             color=plot_colors[i], label=f'E{i+1}')
        
        #threshold line
        ax_v.axhline(y=params['v_thresh'], color='r', linestyle='--', label='Threshold')
        #rest potential line
        ax_v.axhline(y=params['v_rest'], color='k', linestyle=':', label='Rest')
        
        ax_v.set_title("Excitatory Neuron Membrane Potentials")
        ax_v.set_xlabel("Time (ms)")
        ax_v.set_ylabel("Membrane Potential (mV)")
        ax_v.set_xlim(0, self.sim_duration)
        ax_v.grid(True)
        ax_v.legend(loc='upper right', ncol=2)
        
        param_text = f"Parameters: v_rest={params['v_rest']:.1f}mV, v_thresh={params['v_thresh']:.1f}mV, τm={params['tau_m']:.1f}ms, τrefrac={params['tau_refrac']:.1f}ms"
        result_fig.suptitle(param_text, fontsize=10)
        
        plt.tight_layout()
        
        self.plot_weights(input_spikes)
        
    def plot_weights(self, input_spikes):
        print("Synaptic weights after learning:")
        all_weights = []
        weight_labels = []
        
        for i, proj in enumerate(self.input_to_excitatory_projections):
            try:
                try:
                    weights = proj.get('weight')
                except TypeError:
                    try:
                        weights = proj.get('weight', format='list')
                    except:
                        weights = proj.get('weight', format='array')
                
                print(f"Input {i+1} to Excitatory weights: {weights}")
                
                if len(input_spikes[i]) > 0:
                    if weights is not None and len(weights) > 0:
                        if isinstance(weights, list):
                            if isinstance(weights[0], tuple):
                                w_array = np.array([w for _, _, w in weights])
                            else:
                                w_array = np.array(weights)
                        else:
                            w_array = np.array(weights).flatten()
                        
                        all_weights.append(w_array)
                        weight_labels.append(f"Input {i+1}")
            except Exception as e:
                print(f"Couldn't retrieve weights for Input {i+1}: {str(e)}")
        
        try:
            if all_weights:
                plt.figure("Synaptic Weights")
                plt.boxplot(all_weights, labels=weight_labels)
                plt.title("Synaptic Weights after Learning")
                plt.xlabel("Input")
                plt.ylabel("Weight")
                plt.grid(True)
                
                plt.show(block=False)
                
                print("Weight visualization created successfully!")
            else:
                print("No weights available to visualize")
            
        except Exception as e:
            print(f"Error creating weight visualization: {str(e)}")
            
    def end(self):
        """Clean up the simulation"""
        try:
            self.sim.end()
        except:
            pass
        print("Simulation complete.")

def run_simulation(params, input_spikes, colors, duration):
    """
    Run a full neural network simulation
    
    Parameters:
    params (dict): Dictionary of neuron parameters
    input_spikes (list): List of spike times for each input
    colors (list): List of colors for visualization
    duration (float): Duration of the simulation in ms
    
    Returns:
    bool: Whether the simulation was successful
    """
    try:
        simulation = NeuralSimulation()
        simulation.setup(params, input_spikes, duration)
        simulation.run()
        simulation.plot_results(params, input_spikes, colors)
        simulation.end()
        
        return True
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            simulation.end()
        except:
            pass
        return False
