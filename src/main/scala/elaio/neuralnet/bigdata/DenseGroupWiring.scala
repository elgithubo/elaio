package elaio.neuralnet.bigdata

object DenseGroupWiring extends AdditionalWiring {
  override def wire(context: AdditionalWiring.Context): Unit =
    for {
      sourceGroupIndex <- context.groups.indices
      targetGroupIndex <- sourceGroupIndex + 1 until context.groups.length
      source <- context.groups(sourceGroupIndex).neurons
      target <- context.groups(targetGroupIndex).neurons
      if connectionAllowed(source, target)
    } context.connect(source, target)
}
