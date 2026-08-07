package elaio.neuralnet.activation

object Activation {

  // leaky ReLU leak factor. Must be large to avoid vanishing gradients in deep networks.
  // rank of the 6->6 map: 2 at leak 0.01, 6 at 0.8 
  private val leakyReluLeak: Double = 0.8d

  private val squareScale: Double = 1000d

  // Leaky ReLU. The leak must be large: one layer has condition number 1/leak and
  // depth multiplies that out.
  def activationFunctionLeakyRelu(input: Double): Double = {
    if (input > 0d) input else leakyReluLeak * input
  }

  // derivative of activationFunctionLeakyRelu
  def backpropagationFunctionLeakyRelu(input: Double): Double = {
    if (input > 0d) 1d else leakyReluLeak
  }

  // Simple scaled input² activation
  def activationFunctionSquare(input: Double): Double = {
    input * input / squareScale
  }

  // derivative of activationFunctionSquare
  def backpropagationFunctionSquare(input: Double): Double = {
    2d * input / squareScale
  }

  // E[activation^2] / Var(input), used by WeightInitializer to keep signal variance stable
  def secondMomentFactor: Double = (1d + leakyReluLeak * leakyReluLeak) / 2d
}
