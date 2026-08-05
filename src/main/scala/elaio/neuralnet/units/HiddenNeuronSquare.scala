package elaio.neuralnet.units

import elaio.neuralnet.activation.Activation
import elaio.neuralnet.units.Neuron


class HiddenNeuronSquare extends Neuron {

  override def activationFunction(input: Double): Double = 
    Activation.activationFunctionSquare(input)
  override def activationDerivative(input: Double): Double =
    Activation.backpropagationFunctionSquare(input)

}
