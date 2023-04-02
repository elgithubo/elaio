package de.elaio.neuralnet.units

import de.elaio.neuralnet.trace.NetTrace

class InputNeuron extends Neuron {

  NetTrace.WriteMessage(
    "  creating input neuron "
  )

  override def collectInConnections(): Double = {
    _value
  }    
}
