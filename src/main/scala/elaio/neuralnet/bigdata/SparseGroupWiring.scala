package elaio.neuralnet.bigdata

object SparseGroupWiring extends AdditionalWiring {
  override def wire(context: AdditionalWiring.Context): Unit =
    for {
      sourceGroupIndex <- context.groups.indices
      targetGroupIndex <- sourceGroupIndex + 1 until context.groups.length
      sourceGroup = context.groups(sourceGroupIndex)
      targetGroup = context.groups(targetGroupIndex)
      sourceIndex <- sourceGroup.neurons.indices
      targetIndex <- neighboringTargetIndices(sourceIndex, sourceGroup.neurons.length, targetGroup.neurons.length)
      source = sourceGroup.neurons(sourceIndex)
      target = targetGroup.neurons(targetIndex)
      if connectionAllowed(source, target)
    } context.connect(source, target)

  private def neighboringTargetIndices(
      sourceIndex: Int,
      sourceWidth: Int,
      targetWidth: Int
  ): IndexedSeq[Int] = {
    val center =
      if (sourceWidth == 1) targetWidth / 2
      else math.round(sourceIndex.toDouble * (targetWidth - 1) / (sourceWidth - 1)).toInt
    (center - 1 to center + 1).filter(targetIndex => targetIndex >= 0 && targetIndex < targetWidth)
  }
}
