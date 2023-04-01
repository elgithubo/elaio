package de.elaio.neuralnet.units

import de.elaio.neuralnet.bigdata.container.DataCreator

class NeuronDataCreator extends DataCreator {

  override def create( neuronType: NeuronType.Value ): Neuron = {
    if (neuronType == NeuronType.Input)
      createInput()
    else if (neuronType == NeuronType.Output)
      createOutput()
    else
      createHidden()
  }
  override def createInput( ): Neuron = {
    new InputNeuron
  }
  override def createHidden( ): Neuron = {
    new HiddenNeuron
  }
  override def createOutput( ): Neuron = {
    new OutputNeuron
  }
}
