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
  //
  // The leak has to be large. This activation's own derivative is 1 on one side
  // and leak on the other, so a single layer has condition number 1/leak, and
  // the eleven layers between input and output multiply that out. At the usual
  // 0.01 that reaches 1e22 and collapses the input->output map to rank 2 of 6,
  // which no weighting scheme can undo - measured: rank 2.00 at leak 0.01, 4.00
  // at 0.1, 6.00 at 0.8. A leak of 1 would be perfectly conditioned but linear,
  // and could then only ever represent linear maps.
  private val leak: Double = 0.8d

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
