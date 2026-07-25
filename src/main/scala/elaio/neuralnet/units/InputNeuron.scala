package elaio.neuralnet.units

class InputNeuron extends Neuron {

  // an input neuron holds its value instead of computing one, so it ignores
  // its (empty) in-connections rather than summing over them
  override def collectInConnections(): Double = {
    _value
  }

  def initInput(initValue: Double): Unit = {
    _value = initValue
  }
}
