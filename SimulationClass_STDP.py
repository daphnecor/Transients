# dependencies
import numpy as np
import nest
import nest.raster_plot
import itertools
import matplotlib.pyplot as plt


class UpDownPatterns:
    
    def __init__(self, sim_params, neuron_params, syn_params_ex, syn_params_in, STDP='None'):

        self.STDP = STDP
        self.N_total = sim_params.get('N_total')
        self.NE = sim_params.get('NE')
        self.NI = sim_params.get('NI')
        self.J_ex = sim_params.get('J_ex')
        self.J_in = sim_params.get('J_in')
        self.eps = sim_params.get('eps')

        self.resolution = sim_params.get('resolution')
        self.delay = sim_params.get('delay')

        self.n_threads = sim_params.get('n_threads')
        self.stim_start = sim_params.get('stim_start')
        self.stim_end = sim_params.get('stim_end')
        self.simtime = sim_params.get('simtime')
        self.sub_fr = sim_params.get('sub_fr')
        self.sup_fr = sim_params.get('sup_fr')
        
        # set neuron parameters
        self.neuron_params = neuron_params
        
        # set synapse parameters
        self.syn_params_ex = syn_params_ex
        self.syn_params_in = syn_params_in

        self.set_stim_amps()  # input current corresponding to 0/1 in pattern

    def set_stim_amps(self):
        '''adjust the stimulation amplitudes to be slightly sub and supra threshold.'''
        I_th = (self.neuron_params['V_th'] - self.neuron_params['E_L']) * self.neuron_params['g_L']
        self.stim_amps = np.array([self.sub_fr, self.sup_fr]) * I_th
        self.Asub = self.stim_amps[0]
        self.Asupra = self.stim_amps[1]
        
      
    def simulate(self, idx, pattern):
        
        # ====== RESET =========
        #TODO check if its better to use ResetNetwork() here 
        nest.ResetKernel() # gets rid of all nodes, customised models and resets internal clock to 0 
        #nest.ResetNetwork()
        nest.SetKernelStatus({'resolution': self.resolution, 'print_time': False, 'local_num_threads': self.n_threads})
            
        print(idx)
        
        # ====== MAKE NEURON POPULATIONS =========
        group_size = int((self.N_total / len(pattern)))
        SUB_pop = []
        SUPRA_pop = []
        
        for i in range(len(pattern)):
        
            # set defaults for the neuron populations
            nest.SetDefaults('iaf_cond_alpha', self.neuron_params)
                
            if pattern[i] == 1:
                # create supra population of given group size
                n_supra = nest.Create('iaf_cond_alpha', group_size)
                SUPRA_pop.append(n_supra)
            
            elif pattern[i] == 0:
                # create sub population of given group size 
                n_sub = nest.Create('iaf_cond_alpha', group_size)
                SUB_pop.append(n_sub)
        
        # convert to one tuple because NEST likes that
        neurons_supra = tuple([item for sublist in SUPRA_pop for item in sublist])
        neurons_sub = tuple([item for sublist in SUB_pop for item in sublist])
        
        # combine neurons into one big population (for spikedetector and multimeter)
        neurons_all = neurons_supra + neurons_sub
        
        # ====== CREATE DC GENERATORS + SPIKE DETECT + MULTMETER =========
        # create two independent dc generators
        dcgen_sub = nest.Create('dc_generator', params={'amplitude': self.Asub, 'start':self.stim_start, 'stop':self.stim_end})
        dcgen_supra = nest.Create('dc_generator', params={'amplitude': self.Asupra, 'start':self.stim_start, 'stop':self.stim_end})
        
        # create spikedetector
        spikedet = nest.Create('spike_detector')
        # create multimeter that records the voltage
        multimet = nest.Create('multimeter', params={'record_from': ['V_m']})

        # set status voltage meter with a recording interval 
        nest.SetStatus(multimet, params={'interval':1.})
        
        # set status spikedetector
        nest.SetStatus(spikedet, params={"withgid": True, "withtime": True})
        
        # ====== PARAMETERISE SYNAPSES AND CONNECT NEURONS =========
     
        # define network connectivity
        conn_dict = {'rule': 'pairwise_bernoulli', 'p': self.eps} 
        
        static_ex_params = {'model':'static_synapse','weight': self.J_ex, 'delay': self.delay}
        static_in_params = {'model':'static_synapse','weight': self.J_in, 'delay': self.delay}
        
        '''
        For the creation of custom synapse types from already existing synapse types, the command CopyModel is used. 
        '''
        
        if self.STDP == 'ALL':
          
            # make connections between the two populations
            # from exc neurons to all neurons
            nest.Connect(neurons_all[:self.NE], neurons_all, conn_dict, self.syn_params_ex)
            # from interneurons to all neurons
            nest.Connect(neurons_all[self.NE:], neurons_all, conn_dict, self.syn_params_in)
   
        elif self.STDP == 'EXC':
         
            # keep the inhibitory synapses static
            #nest.CopyModel(existing='static_synapse', new='syn_in', params={'weight':self.J_in, 'delay': self.delay})
            
            # make connections between the two populations
            # from exc neurons to all neurons
            nest.Connect(neurons_all[:self.NE], neurons_all, conn_dict, self.syn_params_ex)
            # from interneurons to all neurons
            nest.Connect(neurons_all[self.NE:], neurons_all, conn_dict, static_in_params)
        
        elif self.STDP == 'INH':
            
            # keep the excitatory synapses static
            #nest.CopyModel(existing='static_synapse', new='syn_ex', params={'weight':self.J_ex, 'delay': self.delay})
            
            # make connections between the two populations
            # from exc neurons to all neurons
            nest.Connect(neurons_all[:self.NE], neurons_all, conn_dict, static_ex_params)
            # from interneurons to all neurons
            nest.Connect(neurons_all[self.NE:], neurons_all, conn_dict, self.syn_params_in)
        
        else:
            # synapses will be static (no change in weights)
            nest.CopyModel(existing='static_synapse', new='syn_ex', params={'weight':self.J_ex, 'delay': self.delay})
            nest.CopyModel(existing='static_synapse', new='syn_in', params={'weight':self.J_in, 'delay': self.delay})

            # make connections between the two populations
            nest.Connect(neurons_all[:self.NE], neurons_all, conn_spec=conn_dict, syn_spec='syn_ex' )
            nest.Connect(neurons_all[self.NE:], neurons_all, conn_spec=conn_dict, syn_spec='syn_in' )
        
        # get network connectivity matrix
        M = nest.GetConnections()
        
        # ====== CONNECT TO DEVICES =========
        nest.Connect(neurons_all, spikedet)
        nest.Connect(multimet, neurons_all)
        
        # connect dc_generators to neuron populatioddns
        nest.Connect(dcgen_sub, neurons_sub)
        nest.Connect(dcgen_supra, neurons_supra)

        # ====== SIMULATE =========
        # simulate for a certain time period (ms)
        nest.Simulate(self.simtime)
        
        # spike detector data
        spike_times = nest.GetStatus(spikedet, 'events')[0]['times']
        spike_neurons = nest.GetStatus(spikedet, 'events')[0]['senders']
        
        # multimeter data
        events = nest.GetStatus(multimet)[0]['events']
        etimes = events['times']
        

        return M, spikedet, multimet, events, etimes, spike_times, spike_neurons
    
    
class HelperFuncs:
    '''
    useful helper functions
    '''
    
    def list2str(self, list):
        mystr = ''
        mystr = mystr.join([str(x) for x in list])
        return mystr
    

    def make_permutations(self, seed=[0,0,0,0,1,1,1,1]):
        '''Returns all permutations for given binary list'''
        
        for comb in itertools.combinations(range(len(seed)), seed.count(1)):
            result = [0] * len(seed)
            for i in comb:
                result[i] = 1
            yield result
        return 
    
    
    def get_transient_data(self, spike_times_arr, spike_neurons_arr, sim_params):
        '''
        Uses the information from the simulation run to get information about the transient.
        '''
        transient_spikes = []
        transient_lifetime = []
        transient_uniquen = []
        transient_time_arr = []

        stim_end = sim_params['stim_end']

        for trial in range(len(spike_times_arr)):

            # select data from experiment / trial
            times = spike_times_arr[trial]
            neurons = spike_neurons_arr[trial]

            # === number of spikes ===
            num_trans_spikes = sum(1*(times > stim_end))
            transient_spikes.append(num_trans_spikes)

            # === transient lifetime ===
            transient_time = times[times > stim_end]
            transient_time_arr.append(transient_time)

            if num_trans_spikes == 0:
                transient_lifetime.append(0)
            else:
                t_dur = round(max(transient_time_arr[trial]) - stim_end,2)
                transient_lifetime.append(t_dur)

            # === transient_size ===
            transients = 1*(times > stim_end)
            transient_indices = np.argwhere(transients)
            # take the neurons from these indices
            active_neurons = np.unique(neurons[transient_indices])
            transient_uniquen.append(active_neurons)

        transient_size = [len(i) for i in transient_uniquen]

        return transient_spikes, transient_lifetime, transient_size

    def plot_statevars(self, spike_times_arr, spike_neurons_arr, times_lst, events_lst, idx, stim_time=50):
        '''
        plot the dynamic state variables & raster plots
        '''

        # ==== CHOOSE AN INDEX ====
        fig, (ax1, ax2) = plt.subplots(2, figsize=(22,10))
        #fig.suptitle(f'Pattern: {H.list2str(patterns[idx])}', size=15)

        # raster plot
        ax1.scatter(spike_times_arr[idx], spike_neurons_arr[idx], marker='o', s=0.05, color='k');
        ax1.set_ylabel('neuron id')
        ax1.axvline(x=0, linewidth=2.5, color='xkcd:dark blue', linestyle='--')
        ax1.axvline(x=stim_time, linewidth=2.5, color='xkcd:dark blue', linestyle='--')
        ax1.set_xlim([-20, 1000])

        # voltage
        ax2.plot(times_lst[idx], events_lst[idx]['V_m'], color='xkcd:salmon')
        ax2.set_ylabel('$V_m$ (mV)')
        ax2.axvline(x=0, linewidth=2.5, color='xkcd:dark blue', linestyle='--')
        ax2.axvline(x=stim_time, linewidth=2.5, color='xkcd:dark blue', linestyle='--')
        #ax2.axis([-20,500,-90,-20])
        ax2.set_xlim([-20, 1000])
        
    def plot(self, transient_mins, transient_means, transient_maxs, x_axis_name, parameter_range):
        '''
        takes in a list of transient means and max values, name for the x_axis (the parameter value that is being changed) 
        and the range of parameter values used.

        Plots the graphs.
        '''
        # [mean transient size, mean transient lifetime]
        trans_size_mins = [i[0] for i in transient_mins]
        trans_lifetime_mins = [i[1] for i in transient_mins]

        trans_size_means = [i[0] for i in transient_means]
        trans_lifetime_means = [i[1] for i in transient_means]    

        trans_size_maxs = [item[0] for item in transient_maxs]
        trans_lifetime_maxs = [item[1] for item in transient_maxs]

        # === Plot ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.plot(parameter_range, trans_size_mins, '--go', label='min transient size');
        ax1.plot(parameter_range, trans_size_maxs, '--ro', label ='max transient size');
        ax1.plot(parameter_range, trans_size_means, '--bo', label='mean transient size');
        ax1.set_xlabel(x_axis_name)
        ax1.set_ylabel('transient size (neurons)')
        ax1.legend();

        ax2.plot(parameter_range, trans_lifetime_mins, '--go', label ='min transient times');
        ax2.plot(parameter_range, trans_lifetime_maxs, '--ro', label ='max transient times');
        ax2.plot(parameter_range, trans_lifetime_means, '--bo', label='mean transient times');
        ax2.set_xlabel(x_axis_name);
        ax2.set_ylabel('transient lifetime (ms)')
        ax2.legend();