# import dependencies
import numpy as np
import nest
import nest.raster_plot
import itertools

'''
LIF neurons simulation class.
'''

class LeakySimulator:
    
    def __init__(self, sim_params, model_params, syn_params_ex, syn_params_in):
        
        self.sim_params = sim_params
        self.model_params = model_params
        self.syn_params_ex = syn_params_ex
        self.syn_params_in = syn_params_in

        self.set_stim_amps() 

    def set_stim_amps(self):
        '''adjust the stimulation amplitudes to be slightly sub and supra threshold.'''
        I_th = (self.model_params['V_th'] - self.model_params['E_L']) * self.model_params['g_L']
        self.stim_amps = np.array([self.sim_params['sub_fr'], self.sim_params['sup_fr']]) * I_th
        #print(self.stim_amps)
        
    def build_network(self, STDP='NO'):
        '''
        Build the network architecture.
        '''
        nest.ResetKernel() 
        nest.SetKernelStatus({'resolution': self.sim_params['resolution'], 'print_time': False, \
                              'local_num_threads':self.sim_params['n_threads']})
        
        # ====== CREATE NEURONS =========
        self.neuron_ids = nest.Create(model='iaf_cond_alpha', n=self.sim_params['N_total'], params=self.model_params)

        #  ====== CREATE & CONFIGURE GENERATORS + DETECTORS =========
        # create & configure dc generators (inputs)
        nest.SetDefaults('dc_generator', {'start': self.sim_params['stim_start'], 'stop': self.sim_params['stim_end']}) 
        self.dcgens = nest.Create('dc_generator', self.sim_params['N_total']) # len(dcgens) == len(neuron_ids)

        # create and configure spikedetector
        self.spikedet = nest.Create('spike_detector')
        nest.SetStatus(self.spikedet, params={"withgid": True, "withtime": True})

        # create and configure multimeter that records the voltage (V_m)
        self.multimet = nest.Create('multimeter', params={'record_from': ['V_m']})
        nest.SetStatus(self.multimet, params={'interval':1.})
        
        # ====== CONNECT NEURONS TO DEVICES =========
        nest.Connect(self.neuron_ids, self.spikedet)
        nest.Connect(self.multimet, self.neuron_ids)
        nest.Connect(self.dcgens, self.neuron_ids, 'one_to_one')
        
        # ====== DEFINE SYNAPSES & CONNECTIVITY RULE =========
        static_ex_params = {'model':'static_synapse','weight':self.sim_params['J_ex'], 'delay':self.sim_params['delay']}
        static_in_params = {'model':'static_synapse','weight':self.sim_params['J_in'], 'delay':self.sim_params['delay']}

        conn_dict = {'rule': 'pairwise_bernoulli', 'p': self.sim_params['eps']} 

        # only use STDP if explicity mentioned, by default static synapses
        if STDP == 'exc_only':
             
            # from excitatory neurons to all neurons
            nest.Connect(self.neuron_ids[:self.sim_params['NE']], self.neuron_ids, conn_dict, self.syn_params_ex)
            # from interneurons to all neurons
            nest.Connect(self.neuron_ids[self.sim_params['NE']:], self.neuron_ids, conn_dict, static_in_params)
            
        elif STDP == 'inh_only':
            pass
            
          
        elif STDP == 'all':
            
            syn_STDP = {'model':'stdp_synapse'}
            
            # from excitatory neurons to all neurons
            nest.Connect(self.neuron_ids[:self.sim_params['NE']], self.neuron_ids, conn_dict, self.syn_params_ex)
            # from interneurons to all neurons
            nest.Connect(self.neuron_ids[self.sim_params['NE']:], self.neuron_ids, conn_dict, self.syn_params_in)
            
        else:    
            # ===== CONNECT NEURONS ======
            nest.Connect(self.neuron_ids[:self.sim_params['NE']], self.neuron_ids, conn_dict, static_ex_params)
            nest.Connect(self.neuron_ids[self.sim_params['NE']:], self.neuron_ids, conn_dict, static_in_params)
        
        
    def set_pattern(self, pattern):
        
        '''
        Sets the amplitudes (sub or supra threshold) based on the 8 digit pattern that is given.
        '''
        chunk_size = int(self.sim_params['N_total']/len(pattern)) 
        indices = list(range(self.sim_params['N_total'])) 

        # divide neuron indices in groups of chunk_size
        make_n_chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

        for i in range(len(pattern)):

            stims_i = make_n_chunks[i]
            bit = pattern[i] # >>> 0 or 1
            nest.SetStatus( list(np.array(self.dcgens)[stims_i]), params={'amplitude': self.stim_amps[bit]} )
         
    
    def simulate(self):

        nest.ResetNetwork() # forget all previous simulation data
        nest.SetKernelStatus({'time':0.}) # turn back the clock
        nest.Simulate(1000.) # simulate for a certain time period (ms)
        
        # === GET DATA ===
        spike_times = nest.GetStatus(self.spikedet, 'events')[0]['times']
        spike_neurons = nest.GetStatus(self.spikedet, 'events')[0]['senders']
       
        events = nest.GetStatus(self.multimet)[0]['events']
        etimes = events['times']
        
        return self.spikedet, self.multimet, spike_times, spike_neurons, events, etimes


'''
Other functions that are useful.
'''

class Usefulfunctions:
    
    
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
    
    def circshift(self, V, offset=0):
       
        l = len(V)
        if type(V) is list:
            return V[-offset%l:]+V[:-offset%l]
        elif type(V) is np.ndarray:
            return list(V[-offset%l:])+list(V[:-offset%l])
    
    def reorder(self, neuronids, numneurons, numgroups=8, offset=1):
        
        neworder = []
        # Per group neuron counts
        Gr = np.array([len(np.where(np.arange(numneurons)%numgroups==n)[0]) for n in range(numgroups-1)])
        # CS_Gr = cumsum(circshift(hstack([0,Gr]),-1))
        Gr = np.cumsum(np.hstack([0,Gr]))
        Gr = np.array(self.circshift(Gr,offset))
        for s in neuronids:
            o = s // numgroups
            g = s % numgroups
            #print(int(g)+into)
            neworder.append(Gr[int(g)] + o)
            # neworder.append(int(sum(Gr[0:g]) + o))
            # group.append(g)
        # print "last neuron id to fire (reordered): "+str(max(array(neworder)))
        return neworder
    
    
    
    
    
    
    
    
        
