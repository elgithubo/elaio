package elaio.neuralnet.units

import elaio.neuralnet.activation.Activation

class HiddenNeuronLeakyRelu(id: Long) extends Neuron(id) {
  override def activationFunction(input: Double): Double = {
    Activation.activationFunctionLeakyRelu(input)
  }

  override def activationDerivative(input: Double): Double = {
    Activation.backpropagationFunctionLeakyRelu(input)
  }
}
