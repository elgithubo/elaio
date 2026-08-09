package elaio.neuralnet.connections

final class Weight(var value: Double = 0d) {
  private[neuralnet] var accumulatedGradient: Double = 0d
}
