package elaio.neuralnet.units

class NeuronDataCreator {
  def create(neuronType: NeuronType.Value, id: Long): Neuron = {
    if (neuronType == NeuronType.Input)
      createInput(id)
    else if (neuronType == NeuronType.Output)
      createBackpropagation(id)
    else
      createHidden(id)
  }
  protected def createInput(id: Long): Neuron = {
    new InputNeuron(id)
  }
  protected def createHidden(id: Long): Neuron = {
    new HiddenNeuron(id)
  }
  protected def createBackpropagation(id: Long): Neuron = {
    new OutputNeuron(id)
  }
}
