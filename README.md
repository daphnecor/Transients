# Transients in random recurrent neural networks

This repository contains the code used for the simulation and analysis of networks that operate in a balanced state. We study the emergence of spontaneous activity and the dynamics of balanced (random) recurrent neural networks through computer simulations. By carefully tuning the parameters of a network of conductance based integrate-and-fire neurons we reproduced the transient neural activity as observed in experiments. For the set of parameterisations that pushed the network into a balanced state, transient distributions and spike trains are analysed. Finally, plasticity is introduced in the network by replacing the `static_synapse` with the `stdp_synapse`. 

> You can find my bachelor thesis [here](https://drive.google.com/file/d/1AwQ1LHuOX7oQybp-b4F4ygSo8rua01Qd/view?usp=sharing)

## The Neural Simulation Tool

We use the simulator NEST to implement a network of `iaf_cond_alpha` neurons, which are spiking neurons that use integrate-and-fire (IAF) dynamics with conductance-based synapses.
