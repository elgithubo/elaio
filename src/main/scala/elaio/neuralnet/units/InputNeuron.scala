package elaio.neuralnet.units

import elaio.neuralnet.trace.NetTrace

class InputNeuron extends Neuron {

  NetTrace.WriteMessage(
    "  creating input neuron: " + id
  )

  override def collectInConnections(): Double = {
    _value
  }    
}
