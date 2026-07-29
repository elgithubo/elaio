package elaio.neuralnet.training

import elaio.neuralnet.units.{Neuron, OutputNeuron}

object Backpropagation {
  // One step of gradient descent over the graph's reverse order.
  // Note that a forward pass must have run. 
  //delta is -dL/dz for L = 0.5*(target - value)^2, hence the updates add rather than subtract.
  //
  // collectInConnections averages instead of summing, so the same 1/N appears here:
  //   delta_j = f'(z_j) * sum_k (delta_k * w_jk / N_k)   <- N of the target k
  //   dw_ij   = delta_j * a_i / N_j                      <- N of the owner j
  def run(outputNodes: Array[Neuron], learningRate: Double): Unit = {
    val reverseOrder = GraphTraversal.reverseTopologicalFromOutputs(outputNodes)
    val reachable = reverseOrder.toSet
    val outputSet = outputNodes.toSet

    for (output <- outputNodes) {
      val outputNeuron = output.asInstanceOf[OutputNeuron]
      val derivative = outputNeuron.activationDerivative(outputNeuron.preActivation)
      outputNeuron.delta_((outputNeuron.target - outputNeuron.value) * derivative)
    }

    for (neuron <- reverseOrder.iterator if !outputSet.contains(neuron)) {
      val derivative = neuron.activationDerivative(neuron.preActivation)
      val outgoingSum = neuron.connectionsOut.foldLeft(0d) { (sum, connection) =>
        val targetNeuron = connection.getNeuronTarget
        if (reachable.contains(targetNeuron))
          sum + connection.weight * targetNeuron.delta / targetNeuron.connectionsIn.length
        else sum
      }
      neuron.delta_(outgoingSum * derivative)
    }

    for (neuron <- reverseOrder.reverseIterator) {
      val fanIn = neuron.connectionsIn.length
      for (connectionIn <- neuron.connectionsIn) {
        val sourceValue = connectionIn.getNeuronSource.value
        connectionIn.weight = connectionIn.weight + learningRate * neuron.delta * sourceValue / fanIn
      }
    }
  }
}
