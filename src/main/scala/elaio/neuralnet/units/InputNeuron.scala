package elaio.neuralnet.units

import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.activation.Activation

class InputNeuron(id: Long) extends Neuron(id) {

  // holds its value instead of computing one
  override def collectInConnections(cache: NeuronCollectionCache): Double = {
    _value
  }

  def initInput(initValue: Double): Unit = {
    _value = initValue
  }

  override def activationFunction(input: Double): Double = {
    Activation.activationFunctionLeakyRelu(input)
  }

  override def activationDerivative(input: Double): Double = {
    Activation.backpropagationFunctionLeakyRelu(input)
  }
}
