package elaio.neuralnet.training

import elaio.neuralnet.processing.GraphTraversal
import elaio.neuralnet.units.{Neuron, OutputNeuron}

object Backpropagation {
  // one step of gradient descent over the graph's reverse order.
  // note that a forward pass must have run.
  //
  // delta is -dL/dz for L = 0.5*(target - value)^2, hence the updates add rather than subtract.
  //
  // collectInConnections averages instead of summing, so the same 1/N appears here:
  //   delta_j = f'(z_j) * sum_k (delta_k * w_jk / N_k)   <- N of the target k
  //   dw_ij   = delta_j * a_i / N_j                      <- N of the owner j
  // maxUpdateNorm caps the length of the whole update vector, leaving its direction
  // alone. Measured at initialisation, that length varies by a factor of ~89 between
  // the median step and the worst one, and it is the extreme steps that blow the net
  // up in the first few epochs. Scaling the whole vector rather than each weight on
  // its own keeps the step pointing along the gradient.
  def run(outputNodes: Array[Neuron], learningRate: Double,
          maxUpdateNorm: Double = Double.PositiveInfinity): Unit = {
    val order = GraphTraversal.reverseTopologicalFromOutputs(outputNodes)

    for (output <- outputNodes) {
      val outputNeuron = output.asInstanceOf[OutputNeuron]
      outputNeuron.delta =
        (outputNeuron.target - outputNeuron.value) * outputNeuron.activationDerivative(outputNeuron.preActivation)
    }

    for (neuron <- order.sequence.iterator if !order.outputs.contains(neuron))
      neuron.delta =
        neuron.connectionsOut.foldLeft(0d) { (sum, connection) =>
          val targetNeuron = connection.neuronTarget
          if (order.reachable.contains(targetNeuron))
            sum + connection.weight * targetNeuron.delta / targetNeuron.connectionsIn.length
          else sum
        } * neuron.activationDerivative(neuron.preActivation) // outgoing sum * activation derivative

    val scale =
      if (maxUpdateNorm.isPosInfinity) 1d
      else {
        var sumSquares = 0d
        for (neuron <- order.sequence.reverseIterator) {
          val fanIn = neuron.connectionsIn.length
          for (connectionIn <- neuron.connectionsIn) {
            val gradient = neuron.delta * connectionIn.neuronSource.value / fanIn
            sumSquares += gradient * gradient
          }
          // the bias gradient belongs in the norm too, otherwise the cap scales the
          // weights but not the biases and distorts the very direction it preserves
          if (fanIn > 0) sumSquares += neuron.delta * neuron.delta
        }
        val norm = math.sqrt(sumSquares)
        if (norm > maxUpdateNorm) maxUpdateNorm / norm else 1d
      }

    for (neuron <- order.sequence.reverseIterator) {
      val fanIn = neuron.connectionsIn.length
      for (connectionIn <- neuron.connectionsIn)
        connectionIn.weight =
          connectionIn.weight + learningRate * scale * neuron.delta * connectionIn.neuronSource.value / fanIn
      // dz/db = 1, so no 1/N here - the bias is not averaged in the forward pass.
      // Skipped for input neurons, which override collectInConnections and never
      // read their bias, so updating it would only let it drift.
      if (fanIn > 0)
        neuron.bias = neuron.bias + learningRate * scale * neuron.delta
    }
  }
}
