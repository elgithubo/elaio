package elaio.neuralnet.training

import elaio.neuralnet.units.{Neuron, OutputNeuron}
import elaio.neuralnet.processing.NeuronCollectionCache

object Backpropagation {
  // TODO: make the comment clearer, it still is from claude
  // Requires a forward pass (backpropagation=false) to have already run over
  // outputNodes with NeuronCollectionCache populated, so every visited neuron
  // has a fresh value/preActivation. delta here is -dL/dz for L = 0.5*(target-value)^2,
  // so weight updates are additive. The order neurons were cached in during the
  // forward pass is already a valid topological order, so its reverse is a valid
  // order for the backward pass - no separate graph walk needed.
  def run(outputNodes: Array[Neuron], learningRate: Double): Unit = {
    for (output <- outputNodes) {
      val outputNeuron = output.asInstanceOf[OutputNeuron]
      val derivative = outputNeuron.activationDerivative(outputNeuron.preActivation)
      outputNeuron.delta_((outputNeuron.target - outputNeuron.value) * derivative)
    }

    for (neuron <- NeuronCollectionCache.visitedInOrder.reverseIterator) {
      val derivative = neuron.activationDerivative(neuron.preActivation)
      val outgoingSum = neuron.connectionsOut.foldLeft(0d)((sum, connection) =>
        sum + connection.weight * connection.getNeuronTarget.delta
      )
      neuron.delta_(outgoingSum * derivative)
    }

    for (neuron <- outputNodes.iterator ++ NeuronCollectionCache.visitedInOrder.iterator) {
      for (connectionIn <- neuron.connectionsIn) {
        val sourceValue = connectionIn.getNeuronSource.value
        connectionIn.weight = connectionIn.weight + learningRate * neuron.delta * sourceValue
      }
    }
  }
}
