'''
Run experiment with certain parameter settings.
'''
from importlib import reload
import SimulationClass as simu
# update changes made in python script
reload(simu);
# for logging experiments
from sacred import Experiment

# for storing the information
from sacred.observers import MongoObserver

# name the experiment
ex = Experiment('testing_sacred')
# add Mongodb observer
ex.observers.append(MongoObserver())


# configure the experiment. These are the parameters used for running the experiment.

@ex.config
def cfg():
    
    # general simulation parameters
    sim_params = {

        'N_total':1200,
        'J_ex':6.,
        'J_in':-95.,
        'eps':0.1,  #connection probability
        'resolution':0.1,  # temporal resolution of simulation in ms. Kumar2008: 0.1
        'delay':1.5,  # synaptic delay in the network
        'n_threads':8,
        'stim_start':0., # start applying current (dc)
        'stim_end':150., # end applying current (dc)
        'simtime':1000., # simulation time 
        'sub_fr':0.9, # subthreshold current amplitude
        'sup_fr':1.01, # suprathreshold current amplitude
    }

    # set neuron parameters
    neuron_params = {
        'C_m': 250.0,
        'E_L': -70.0,
        'E_ex': 0.0,
        'E_in': -80.0,
        'I_e': 0.0,
        'V_reset': -70.0,
        'V_th': -50.0,
        'g_L': 16.7,
        't_ref': 2.0,
        'tau_syn_ex': 0.326,
        'tau_syn_in': 0.326,
    }
    
# 
    
@ex.automain
def run(sim_params, neuron_params):

    # initialise classes
    udp = simu.UpDownPatterns(sim_params, neuron_params)
    H = simu.HelperFuncs()
    
    # make lists
    spike_times_lst = []
    spike_neurons_lst = []
    multimeters = []
    spikedetectors = []
    
    patterns = []
    # make permutations for 8 bit pattern
    for perm in H.make_permutations():
        patterns.append(perm)
        
        
    # run simulations for experiment
    for i, pattern in enumerate(patterns):
        label = H.list2str(pattern)
        multimet, spikedet, spike_times, spike_neurons = udp.simulate(pattern)

        # convert to python lists 
        spike_times.tolist()
        spike_neurons.tolist()

        spike_times_lst.append(spike_times)
        spike_neurons_lst.append(spike_neurons)
        multimeters.append(multimet)
        spikedetectors.append(spikedet)
    
    return spike_times_lst, spike_neurons_lst
    

