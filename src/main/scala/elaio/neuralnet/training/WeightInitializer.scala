package elaio.neuralnet.training

import elaio.neuralnet.activation.Activation
import elaio.neuralnet.processing.GraphTraversal

object WeightInitializer {
  // He-style init, inverted because collectInConnections averages rather than sums
  def initialize(order: GraphTraversal.ReverseOrder): Long = {
    val random = new scala.util.Random
    var connectionsInitialized = 0L

    for (neuron <- order.sequence)
      if (neuron.connectionsIn.length > 0) {
        val deviation = math.sqrt(neuron.connectionsIn.length / Activation.secondMomentFactor)
        for (connectionIn <- neuron.connectionsIn) {
          connectionIn.weight = random.nextGaussian() * deviation
          connectionsInitialized = connectionsInitialized + 1
        }
      }

    connectionsInitialized
  }
}
