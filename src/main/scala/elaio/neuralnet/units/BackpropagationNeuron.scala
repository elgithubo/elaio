package elaio.neuralnet.units

import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.activation.Activation
import elaio.neuralnet.processing.NeuronCollectionCache

class BackpropagationNeuron() extends Neuron {
  NeuronCollectionCache.clear()
  if(_value > 0)
    collectInConnections(1d, false)
  else
    collectInConnections(0d, false)
}
