package elaio.neuralnet.activation

object Activation {
  def activationFunction(input: Double): Double = {
    if (input > 0) input else 0
  }
  def backpropagationFunction(input: Double): Double = {
    if (input > 0) 1 else 0
  }  
}
