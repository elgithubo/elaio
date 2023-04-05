package elaio.neuralnet.units

import elaio.neuralnet.trace.NetTrace

class InputNeuron extends Neuron {

  NetTrace.WriteMessage(
    "  creating input neuron: " + id
  )

  override def collectInConnections(pullWeight: Double): Double = {
    _value
  }    

  override def init(value: Double, target: Double, tolerance: Double): Unit = {
    _initValue = value
    super.init(value, target, tolerance)
  }

}

