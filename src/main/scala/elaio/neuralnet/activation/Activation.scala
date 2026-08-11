package elaio.neuralnet.activation

object Activation {

  // leaky ReLU leak factor. Must be large to avoid vanishing gradients in deep networks.
  // rank of the 6->6 map: 2 at leak 0.01, 6 at 0.8 
  private val leakyReluLeak: Double = 0.8d

  // Leak of the intermediate output rank. That rank is a single layer feeding a linear read-out,
  // so it pays the 1/leak condition number once instead of compounding it over the depth - it can
  // afford a sharper kink than the hidden ranks, and a sharper kink is what the read-out can use.
  // Sweep range 0.1 - 0.25.
  private val intermediateReluLeak: Double = 0.2d

  private val squareScale: Double = 1000d

  // Leaky ReLU. The leak must be large: one layer has condition number 1/leak and
  // depth multiplies that out.
  def activationFunctionLeakyRelu(input: Double): Double = {
    leakyRelu(input, leakyReluLeak)
  }

  // derivative of activationFunctionLeakyRelu
  def backpropagationFunctionLeakyRelu(input: Double): Double = {
    leakyReluDerivative(input, leakyReluLeak)
  }

  // same kink, sharper leak - see intermediateReluLeak
  def activationFunctionIntermediateRelu(input: Double): Double = {
    leakyRelu(input, intermediateReluLeak)
  }

  // derivative of activationFunctionIntermediateRelu
  def backpropagationFunctionIntermediateRelu(input: Double): Double = {
    leakyReluDerivative(input, intermediateReluLeak)
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

  // one definition of the kink, so a swept leak cannot drift between value and derivative
  private def leakyRelu(input: Double, leak: Double): Double = {
    if (input > 0d) input else leak * input
  }

  private def leakyReluDerivative(input: Double, leak: Double): Double = {
    if (input > 0d) 1d else leak
  }
}
