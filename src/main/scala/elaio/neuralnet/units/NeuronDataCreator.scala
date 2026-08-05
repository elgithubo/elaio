package elaio.neuralnet.units

class NeuronDataCreator {
  def create(neuronType: NeuronType.Value, id: Long): Neuron = {

    if (neuronType == NeuronType.Input)
      createInput(id)
    else if (neuronType == NeuronType.Output)
      createBackpropagation(id)
    else if (neuronType == NeuronType.HiddenSquare)
      createHiddenSquare(id)
    else if (neuronType == NeuronType.HiddenLeakyRelu)
      createHiddenLeakyRelu(id)
    else
      throw new IllegalArgumentException("Unknown neuron type: " + neuronType)
  }
  private def createInput(id: Long): Neuron = {
    new InputNeuron(id)
  }
  private def createHiddenLeakyRelu(id: Long): Neuron = {
    new HiddenNeuronLeakyRelu(id)
  }
  private def createHiddenSquare(id: Long): Neuron = {
    new HiddenNeuronSquare(id)
  }
  private def createBackpropagation(id: Long): Neuron = {
    new OutputNeuron(id)
  }
}
