package elaio.neuralnet.training

import elaio.neuralnet.activation.Activation
import elaio.neuralnet.processing.GraphTraversal
import elaio.neuralnet.units.Neuron

object WeightInitializer {
  // He-style init, inverted because collectInConnections averages rather than sums
  def initialize(outputNodes: Array[Neuron]): Double = {
    val random = new scala.util.Random
    var connectionsInitialized = 0

    for (neuron <- GraphTraversal.reverseTopologicalFromOutputs(outputNodes).sequence)
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
