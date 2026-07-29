package elaio.neuralnet.units

class OutputNeuron() extends Neuron {

  protected var _target: Double = 0d

  // hot path - called per output per example per epoch, so no tracing here
  def initOutput(target: Double): Unit = {
    _target = target
  }

  // linear output: skips the hidden layers' leaky ReLU, which would scale
  // negative targets by leak and positive ones by 1
  override def activationFunction(input: Double): Double = input
  override def activationDerivative(input: Double): Double = 1d

  def target: Double = _target
}
