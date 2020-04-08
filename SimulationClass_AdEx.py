# import dependencies
import numpy as np
import nest
import nest.raster_plot
import itertools


class UpDownPatterns:
    
    def __init__(self, sim_params, model_params):

        self.N_total = sim_params.get('N_total')
        self.NE = sim_params.get('NE')
        self.NI = sim_params.get('NI')
        self.n_threads = sim_params.get('n_threads')
        self.stim_start = sim_params.get('stim_start')
        self.stim_end = sim_params.get('stim_end')
        self.simtime = sim_params.get('simtime')
        self.sub_fr = sim_params.get('sub_fr')
        self.sup_fr = sim_params.get('sup_fr')
        self.resolution = sim_params.get('resolution')
        
        # specific neuron parameters
        self.model_params = model_params
        self.sim_params = sim_params
       
        self.set_stim_amps()  # input current corresponding to 0/1 in pattern
        
        
    def set_stim_amps(self):
        '''adjust the stimulation amplitudes to be slightly sub and supra threshold.'''
        I_th = (self.model_params['V_th'] - self.model_params['E_L']) * self.model_params['g_L']
        self.stim_amps = np.array([self.sub_fr, self.sup_fr]) * I_th
        self.Asub = self.stim_amps[0]
        self.Asupra = self.stim_amps[1]
        
    def simulate(self, pattern):
        
        # ====== RESET =========
        #TODO check if its better to use ResetNetwork() here 
        nest.ResetKernel() # gets rid of all nodes, customised models and resets internal clock to 0 
    
        nest.SetKernelStatus({'resolution': self.resolution, 'print_time': False, 'local_num_threads': self.n_threads})
        
        # ====== MAKE NEURON POPULATIONS =========
        group_size = int((self.N_total / len(pattern)))
        SUB_pop = []
        SUPRA_pop = []
        
        for i in range(len(pattern)):
        
            # parameterise the AdEx neuron model
            nest.SetDefaults('aeif_cond_exp', self.model_params)
                
            if pattern[i] == 1:
                # create supra population of given group size
                n_supra = nest.Create('aeif_cond_exp', group_size)
                
                SUPRA_pop.append(n_supra)
            
            elif pattern[i] == 0:
                # create sub population of given group size 
                n_sub = nest.Create('aeif_cond_exp', group_size)
                
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
        
        # ====== CONNECT NEURONS =========
        nest.CopyModel(existing='static_synapse', new='syn_ex', params={'weight':self.sim_params['J_ex'], 'delay': self.sim_params['delay']})
        nest.CopyModel(existing='static_synapse', new='syn_in', params={'weight':self.sim_params['J_in'], 'delay': self.sim_params['delay']})
        # how we want to connect the neurons
        conn_rule = {'rule': 'pairwise_bernoulli', 'p': 0.1} 
         
        # make connections between the two populations
        nest.Connect(neurons_all[:self.NE], neurons_all, conn_spec=conn_rule, syn_spec='syn_ex' )
        nest.Connect(neurons_all[self.NE:], neurons_all, conn_spec=conn_rule, syn_spec='syn_in' )
        
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
        volt_neuron_ids = nest.GetStatus(multimet, 'events')[0]['senders']
        volt_times = nest.GetStatus(multimet, 'events')[0]['times']
        volt_trace = nest.GetStatus(multimet, 'events')[0]['V_m']

        return multimet, spikedet, spike_times, spike_neurons
    

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