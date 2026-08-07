package elaio.neuralnet.bigdata

import elaio.neuralnet.units.{HiddenNeuronSquare, Neuron}

object DenseGroupWiring extends AdditionalWiring {
  private def connectsSquareNeurons(source: Neuron, target: Neuron): Boolean =
    source.isInstanceOf[HiddenNeuronSquare] && target.isInstanceOf[HiddenNeuronSquare]

  override def wire(context: AdditionalWiring.Context): Unit =
    for {
      sourceGroupIndex <- context.groups.indices
      targetGroupIndex <- sourceGroupIndex + 1 until context.groups.length
      source <- context.groups(sourceGroupIndex).neurons
      target <- context.groups(targetGroupIndex).neurons
      if !connectsSquareNeurons(source, target)
    } context.connect(source, target)
}
