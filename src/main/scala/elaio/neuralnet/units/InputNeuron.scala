package elaio.neuralnet.units

import elaio.neuralnet.trace.NetTrace

class InputNeuron extends Neuron {
  override def collectInConnections(pullWeight: Double): Double = {
    _value
  }    
  override def init(value: Double, target: Double, tolerance: Double): Unit = {
    _initValue = value
    NetTrace.WriteMessage(
      "initializing input neuron - target: " + target
    )
    super.init(value, target, tolerance)
  }
}

