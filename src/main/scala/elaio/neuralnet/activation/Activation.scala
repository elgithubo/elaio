package elaio.neuralnet.activation

object Activation {
  def activationFunction(input: Double): Double = {
    //if (input > 0) input else 0
    return( 1 / ( 1 + math.pow( 1E1, -input ) ) )
  }
  def backpropagationFunction(input: Double): Double = {
    //if (input > 0) 1 else 0
    return( Activation.activationFunction(input) * ( 1 - Activation.activationFunction(input) ) )
  }  
}
