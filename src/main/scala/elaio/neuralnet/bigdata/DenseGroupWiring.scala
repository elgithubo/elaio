package elaio.neuralnet.bigdata

import elaio.neuralnet.units.{HiddenNeuronSquare, InputNeuron, Neuron, OutputNeuron}

object DenseGroupWiring extends AdditionalWiring {
  private def connectsSquareNeurons(source: Neuron, target: Neuron): Boolean =
    source.isInstanceOf[HiddenNeuronSquare] && target.isInstanceOf[HiddenNeuronSquare]
  private def connectsInputToOutput(source: Neuron, target: Neuron): Boolean =
    source.isInstanceOf[InputNeuron] && target.isInstanceOf[OutputNeuron]

  override def wire(context: AdditionalWiring.Context): Unit =
    for {
      sourceGroupIndex <- context.groups.indices
      targetGroupIndex <- sourceGroupIndex + 1 until context.groups.length
      source <- context.groups(sourceGroupIndex).neurons
      target <- context.groups(targetGroupIndex).neurons
      if !connectsSquareNeurons(source, target) && !connectsInputToOutput(source, target)
    } context.connect(source, target)
}
