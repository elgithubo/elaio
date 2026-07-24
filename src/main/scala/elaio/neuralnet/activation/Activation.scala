package elaio.neuralnet.activation

object Activation {
  // Leaky ReLU. The previous logistic squash attenuated every hop by at most
  // its derivative (0.25, and 0.0625 once divided by the old scale factor),
  // which combined with the 1/N averaging in Neuron.collectInConnections wiped
  // out the input signal long before it reached the output. A non-saturating
  // activation has derivative 1 on the positive side, so depth no longer
  // destroys the signal.
  //
  // The negative side leaks instead of returning 0 so a unit can never go
  // permanently dead: a zero derivative would cut off every path running
  // through that neuron, and reaching all neurons is already hard enough in
  // this topology.
  private val leak: Double = 0.01d

  def activationFunction(input: Double): Double = {
    if (input > 0d) input else leak * input
  }

  // derivative of activationFunction
  def backpropagationFunction(input: Double): Double = {
    if (input > 0d) 1d else leak
  }

  // E[activation^2] / Var(input) for zero-mean symmetric input. Used by the
  // weight initialisation to keep the signal variance stable from layer to
  // layer; it lives here so it stays in step with the activation above.
  def secondMomentFactor: Double = (1d + leak * leak) / 2d
}
