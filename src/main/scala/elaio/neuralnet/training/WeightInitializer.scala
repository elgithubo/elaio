package elaio.neuralnet.training

import elaio.neuralnet.activation.Activation
import elaio.neuralnet.units.Neuron

object WeightInitializer {
  // He-style init, inverted because collectInConnections averages rather than sums:
  // Var(w) = N / secondMomentFactor, so weights grow with fan-in instead of shrinking.
  // Runs after the graph is built, walking back from the outputs once fan-in is final.
  def initialize(outputNodes: Array[Neuron]): Int = {
    val random = new scala.util.Random
    var connectionsInitialized = 0

    for (neuron <- GraphTraversal.reverseTopologicalFromOutputs(outputNodes).sequence) {
      val fanIn = neuron.connectionsIn.length
      if (fanIn > 0) {
        val deviation = math.sqrt(fanIn / Activation.secondMomentFactor)
        for (connectionIn <- neuron.connectionsIn) {
          connectionIn.weight = random.nextGaussian() * deviation
          connectionsInitialized = connectionsInitialized + 1
        }
      }
    }
    connectionsInitialized
  }
}
