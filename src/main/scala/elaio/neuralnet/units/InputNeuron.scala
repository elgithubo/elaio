package elaio.neuralnet.units

import elaio.neuralnet.processing.NeuronCollectionCache

class InputNeuron extends Neuron {

  // holds its value instead of computing one
  override def collectInConnections(cache: NeuronCollectionCache): Double = {
    _value
  }

  def initInput(initValue: Double): Unit = {
    _value = initValue
  }
}
