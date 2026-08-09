package elaio.neuralnet.units

import elaio.neuralnet.activation.Activation

class HiddenNeuronSquare(id: Long) extends HiddenNeuron(id) {
  override def activationFunction(input: Double): Double =
    Activation.activationFunctionSquare(input)
  override def activationDerivative(input: Double): Double =
    Activation.backpropagationFunctionSquare(input)
}
