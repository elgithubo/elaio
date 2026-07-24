package elaio.neuralnet.training

import elaio.neuralnet.units.{Neuron, OutputNeuron}
import elaio.neuralnet.processing.NeuronCollectionCache

object Backpropagation {
  // One step of gradient descent over the graph built by the last forward pass.
  //
  // Requires a forward pass (backpropagation = false) to have just run over
  // outputNodes, so every reachable neuron holds a current value/preActivation
  // and NeuronCollectionCache holds the neurons that pass visited.
  //
  // delta is defined as -dL/dz (z being the pre-activation sum) for the loss
  // L = 0.5 * (target - value)^2, which is why the updates below add instead
  // of subtract.
  //
  // collectInConnections averages its incoming values instead of summing them:
  //   z_j = (1 / N_j) * sum_i (w_ij * a_i),  N_j = j's number of in-connections
  // so that same 1/N factor has to appear in the derivatives, otherwise the
  // backward pass computes gradients for a net that sums - a different net:
  //   delta_j = f'(z_j) * sum_k (delta_k * w_jk / N_k)   <- N of the target k
  //   dw_ij   = delta_j * a_i / N_j                      <- N of the owner j
  // Both divisors are >= 1: a connection is always present in its own target's
  // in-connections, and the update loop only runs over non-empty in-connections.
  //
  // A neuron is only cached once all of its own sources are cached, so the cache
  // order is a topological order and its reverse is a valid backward order.
  def run(outputNodes: Array[Neuron], learningRate: Double): Unit = {
    for (output <- outputNodes) {
      val outputNeuron = output.asInstanceOf[OutputNeuron]
      val derivative = outputNeuron.activationDerivative(outputNeuron.preActivation)
      outputNeuron.delta_((outputNeuron.target - outputNeuron.value) * derivative)
    }

    for (neuron <- NeuronCollectionCache.visitedInOrder.reverseIterator) {
      val derivative = neuron.activationDerivative(neuron.preActivation)
      val outgoingSum = neuron.connectionsOut.foldLeft(0d) { (sum, connection) =>
        val targetNeuron = connection.getNeuronTarget
        sum + connection.weight * targetNeuron.delta / targetNeuron.connectionsIn.length
      }
      neuron.delta_(outgoingSum * derivative)
    }

    for (neuron <- outputNodes.iterator ++ NeuronCollectionCache.visitedInOrder.iterator) {
      val fanIn = neuron.connectionsIn.length
      for (connectionIn <- neuron.connectionsIn) {
        val sourceValue = connectionIn.getNeuronSource.value
        connectionIn.weight =
          connectionIn.weight + learningRate * neuron.delta * sourceValue / fanIn
      }
    }
  }
}
