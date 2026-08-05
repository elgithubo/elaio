package elaio.neuralnet.units

class OutputNeuron(id: Long) extends Neuron(id) {

  protected var _target: Double = 0d

  def initOutput(target: Double): Unit = {
    _target = target
  }

  // linear output: skips the hidden layers' leaky ReLU
  override def activationFunction(input: Double): Double = input
  override def activationDerivative(input: Double): Double = 1d

  def target: Double = _target
}
