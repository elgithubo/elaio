package elaio.neuralnet.activation

object Activation {
  // divides the input before the logistic squash so single-digit-scale
  // target values don't all saturate to the same ~1.0 output
  private val scale: Double = 4d

  def activationFunction(input: Double): Double = {
    //if (input > 0) input else 0
    return( 1 / ( 1 + math.exp( -input / scale ) ) )
  }
  // derivative of activationFunction w.r.t. its input, including the 1/scale
  // factor from the chain rule (activationFunction divides by scale internally)
  def backpropagationFunction(input: Double): Double = {
    return( Activation.activationFunction(input) * ( 1 - Activation.activationFunction(input) ) / scale )
  }
}
