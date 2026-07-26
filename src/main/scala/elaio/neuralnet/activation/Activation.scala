package elaio.neuralnet.activation

object Activation {
  // Leaky ReLU. The leak must be large: one layer has condition number 1/leak and
  // depth multiplies that out. Measured rank of the 6->6 map: 2 at leak 0.01, 6 at 0.8.
  private val leak: Double = 0.8d

  def activationFunction(input: Double): Double = {
    if (input > 0d) input else leak * input
  }

  // derivative of activationFunction
  def backpropagationFunction(input: Double): Double = {
    if (input > 0d) 1d else leak
  }

  // E[activation^2] / Var(input), used by WeightInitializer to keep signal variance stable
  def secondMomentFactor: Double = (1d + leak * leak) / 2d
}
