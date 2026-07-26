package elaio.neuralnet.units

class InputNeuron extends Neuron {

  // holds its value instead of computing one
  override def collectInConnections(): Double = {
    _value
  }

  def initInput(initValue: Double): Unit = {
    _value = initValue
  }
}
