package elaio.neuralnet.units

import elaio.neuralnet.trace.NetTrace

class OutputNeuron() extends Neuron {

  protected var _target: Double = 0d

  def initOutput(target: Double): Unit = {
    _target = target
    NetTrace.WriteMessage(
      "initializing output neuron - target: " + target
    )
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
