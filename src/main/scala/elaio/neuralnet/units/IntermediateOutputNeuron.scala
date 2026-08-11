package elaio.neuralnet.units

import elaio.neuralnet.activation.Activation

// expecially desiged for the TensoredContainer output in a stacked LayeredContainer,
// where the output is not the final output of the network but an intermediate output that
// feeds into the LayeredCotnainer's read-out layer.
// It uses the leaky ReLU activation function, which is the same as the hidden layers' activation function.
class IntermediateOutputNeuron(override val id: Long) extends OutputNeuron(id) {
  override def activationFunction(input: Double): Double =
    Activation.activationFunctionLeakyRelu(input)
  override def activationDerivative(input: Double): Double =
    Activation.backpropagationFunctionLeakyRelu(input)
}
