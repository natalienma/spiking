## Equations
### Spike times:
s(t)=k∑​δ(t−tk​)
- list of moments the neuron spiked
- tk = time of the k-th spike
- ​δ, delta = spike shaped function that is 0 everywhere except when fired

### Neuron's voltage activity (in continuous calculus)
τm​dt/dV(t)​=−(V(t)−Vrest​)+RI(t)
- V = voltage, activation
- -(V(t) - Vrest) = bring back to resting voltage
- RI(t) = incoming current pushing voltage up or down
- tau_m = how fast the neuron drains. bigger tau_m = slower draining neuron.

### Neuron's voltage activity (in discrete code)
V[t]=βV[t−1]+I[t]−S[t−1]Vth​
- V[t-1] = voltage at previous time point
- β, beta = a number between 0 and 1, used to decay voltage at previous time 
- βV[t-1]= take the previous voltage and shrink it a bit
- I(t) = the current coming in now
- S[t-1] * Vth = if the neuron just spiked, S[t-1] = 1. Subtract the threshold amount to reset so that it doesn't immediately spike again
**NOTE: on GPUs, the network has to calculate the activity of every neuron at every time step. On event-driven hardware/neuromorphic silicon like Loihi: neuron core only wakes up and computes when it receives a spike.**

### Will this neuron fire? Is it above the threshold?
S[t]=Θ(V[t]−Vth)
- Θ is a function that decides if the voltage passed the threshold. 0 = no, 1 = yes. 

### Total current into a neuron:
I[t]=j∑​wj​Sj​[t]
- j = every upstream neuron. 
- Sj[t] = whether it spiked or not (1 or 0)
- multiply by that synapse's weight, wj

## hyperparameters vs. trained parameters
hyperparameters:
- A+ and A- (strengthen/weaken amounts)
- tau (usually fixed)

trained parameters:
- synapse weights
- threshold (although usually fixed)

## questions:
1. Over time, if B often fires after A, we increase the weight `w` between the two. Isn't this artificially inflating the number of times B will fire? Why not let A naturally influence B?
- `w` is randomly assigned at first. Updating it is just learning

2. why do we overwrite the entire weight matrix with new data every time? and why can we only overwrite 3%? if it was that easy, why haven't we been doing it before?

3. how can delta_t be negative when neuron B firing relies on neuron A to fire? 
- neuron B has many dendrites (inputs), so neuron A can be 0, and neuron B can still fire. 
- an SNN does not look like a typical ANN, where there are rounds, passes, or feed-forward. A and B are their own neurons, like people in a room that shout whenever they hear enough. There are no sequential layers like in ANNs.
- so if B fires before A, delta_t is negative. this just says that A does not cause B.

4. Do the edges/weights have direction?
- Yes. Just because A does not cause B, does not mean B does not cause A. B can cause A if B happens before A and the spikes are close in time.
- for this reason there is w_AB and w_BA, which can have two different values and STDP calculations

5. Infinite Firing Loop? A causes B, B causes A, repeat?
- This recurrent excitation is called a seizure. 
- Inhibitory connections. Some synapses have negative weights
Fix:
- inhibitory connections: some synapses have negative weights that suppress. 


### Why Trace?
Tracing the input neurons' firing behavior is like a stripped down version of a LIF, without the sum. 

You add a `trace[j]` value for each input neuron. 
`trace[j] = beta_trace * trace[j] + (1 if input[j] spiked, else 0)`

### SHD:



## Next steps
1. variable beta for different neurons (heterogeneous time constants)
