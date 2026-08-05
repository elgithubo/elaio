package elaio.neuralnet.units

import elaio.neuralnet.bigdata.container.DataCreator
import elaio.neuralnet.units.HiddenNeuronLeakyRelu

class NeuronDataCreator extends DataCreator {
  override def create( neuronType: NeuronType.Value ): Neuron = {
    if (neuronType == NeuronType.Input)
      createInput()
    else if (neuronType == NeuronType.Output)
      createBackpropagation()
    else if (neuronType == NeuronType.HiddenSquare)
      createHiddenSquare()
    else if (neuronType == NeuronType.HiddenLeakyRelu)
      createHiddenLeakyRelu()
    else
      throw new IllegalArgumentException("Unknown neuron type: " + neuronType)
  }
  override def createInput( ): Neuron = {
    new InputNeuron
  }
  override def createHiddenLeakyRelu( ): Neuron = {
    new HiddenNeuronLeakyRelu
  }
  override def createHiddenSquare( ): Neuron = {
    new HiddenNeuronSquare
  }
  override def createBackpropagation( ): Neuron = {
    new OutputNeuron
  }
}
