package elaio.neuralnet.units

class OutputNeuron() extends Neuron {

  protected var _target: Double = 0d

  // called once per output per example per epoch - keep it free of tracing,
  // a single message here is tens of millions of lines over a training run
  def initOutput(target: Double): Unit = {
    _target = target
  }

  // linear output: targets are arbitrary-magnitude values, not (0,1)-bounded
  // probabilities, so squashing the output through a sigmoid caps what this
  // neuron could ever reconstruct. Hidden layers keep the nonlinearity.
  override def activationFunction(input: Double): Double = input
  override def activationDerivative(input: Double): Double = 1d

  def target: Double =  {
    _target
  }
}
