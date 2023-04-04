package elaio.neuralnet.units

import elaio.neuralnet.bigdata.container.DataCreator

class NeuronDataCreator extends DataCreator {

  override def create( neuronType: NeuronType.Value ): Neuron = {
    if (neuronType == NeuronType.Input)
      createInput()
    else if (neuronType == NeuronType.Backpropagation)
      createBackpropagation()
    else
      createHidden()
  }
  override def createInput( ): Neuron = {
    new InputNeuron
  }
  override def createHidden( ): Neuron = {
    new HiddenNeuron
  }
  override def createBackpropagation( ): Neuron = {
    new BackpropagationNeuron
  }
}
