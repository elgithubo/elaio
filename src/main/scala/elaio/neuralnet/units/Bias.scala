package elaio.neuralnet.units

final class Bias(var value: Double = 0d) {
  private[neuralnet] var accumulatedGradient: Double = 0d
}
