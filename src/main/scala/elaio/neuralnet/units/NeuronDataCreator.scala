package elaio.neuralnet.units

class NeuronDataCreator {
  def create(neuronType: NeuronType.Value, id: Long): Neuron = {

    if (neuronType == NeuronType.Input)
      new InputNeuron(id)
    else if (neuronType == NeuronType.HiddenLeakyRelu)
      new HiddenNeuronLeakyRelu(id)
    else if (neuronType == NeuronType.HiddenSquare)
      new HiddenNeuronSquare(id)
    else if (neuronType == NeuronType.IntermediateOutput)
      new IntermediateOutputNeuron(id)
    else if (neuronType == NeuronType.Output)
      new OutputNeuron(id)         
    else
      throw new IllegalArgumentException("Unknown neuron type: " + neuronType)
  }
}
