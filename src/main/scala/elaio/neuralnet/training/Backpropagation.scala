package elaio.neuralnet.training

import elaio.neuralnet.units.{Neuron, OutputNeuron}

object Backpropagation {
  // One step of gradient descent over the graph's reverse order.
  // Note that a forward pass must have run. 
  //
  // delta is -dL/dz for L = 0.5*(target - value)^2, hence the updates add rather than subtract.
  //
  // collectInConnections averages instead of summing, so the same 1/N appears here:
  //   delta_j = f'(z_j) * sum_k (delta_k * w_jk / N_k)   <- N of the target k
  //   dw_ij   = delta_j * a_i / N_j                      <- N of the owner j
  def run(outputNodes: Array[Neuron], learningRate: Double): Unit = {
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

    for (neuron <- order.sequence.reverseIterator) {
      val fanIn = neuron.connectionsIn.length
      for (connectionIn <- neuron.connectionsIn)
        connectionIn.weight =
          connectionIn.weight + learningRate * neuron.delta * connectionIn.neuronSource.value / fanIn
    }
  }
}
