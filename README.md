The files in this repository are associated with the paper "Hardware-Aware Offline Training of CTT-Based Neuromorphic Hardware".

Two different neuron models are present:
  1. [digLIF](TVLSI26/neuron_models/digLIF.py) A neuron model that exactly matches the digital LIF neuron in the neuromorphic hardware described in this paper." With constants that replicate our Cadence hardware simulations. 
  2. A more general LIF neuron model that could represent other implementations. This was adopted to show the generalizability of the hardware-aware training approach. A variety of LIF implementations exist, and our results showed that LIF neuron hardware implementations require some flexibility in setting the threshold voltage and decay rate to be used for different tasks. Learning would be much easier the closer we get to actual software-based neural networks with many tunable parameters. Otherwise, acquiring an equivalent performance to ANNs requires a lot more architecture exploration.

Two noise injection methodologies are implemented:
  1. [Noise pool generation](TVLSI26/ctt_weights/noisePool.py): sweeps different hardware-related parameters such as temperature, PVT changes, and programming imprecision (based on their unique distribution). Then, it creates an array with all the possible values for different network parameters (representing hardware components).
  2. Sensitivity model: based on analytical calculations of combining the effects of variation in multiple parameters (weights, neuron constants) on the parameter of interest. It first samples a value from a pre-defined distribution for each hardware-based variant and then calculates their combined effect.


