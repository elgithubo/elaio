package elaio.neuralnet.activation

object Activation {
  // factor to divide input before applying the activation function
  // so that the target values don't all saturate to ~1.0d.
  // The value has been determined experimental and seems to fit
  // the net topology regardsless of the buildDim and number of inputs.
  private val scale: Double = 4d

  def activationFunction(input: Double): Double = {
    return( 1 / ( 1 + math.exp( -input / scale ) ) )
  }
  // derivative of activationFunction including the 1/scale factor
  def backpropagationFunction(input: Double): Double = {
    return( Activation.activationFunction(input) * ( 1 - Activation.activationFunction(input) ) / scale )
  }
}
