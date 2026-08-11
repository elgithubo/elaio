package elaio.neuralnet.units

import elaio.neuralnet.activation.Activation

// especially designed for the TensoredContainer output in a stacked LayeredContainer,
// where the output is not the final output of the network but an intermediate output that
// feeds into the LayeredContainer's read-out layer.
//
// It is leaky ReLU rather than the identity of a real output: a linear rank in front of the linear
// read-out would compose to one affine map, so the rank would cost parameters and gradient dilution
// without widening the function class. The leak is the sharper intermediateReluLeak, not the hidden
// ranks' one - this rank pays the 1/leak condition number once, and the read-out needs a kink it
// can actually see.
class IntermediateOutputNeuron(id: Long) extends OutputNeuron(id) {
  override def activationFunction(input: Double): Double =
    Activation.activationFunctionIntermediateRelu(input)
  override def activationDerivative(input: Double): Double =
    Activation.backpropagationFunctionIntermediateRelu(input)
}
