package elaio.neuralnet.training

import scala.collection.mutable
import elaio.neuralnet.activation.Activation
import elaio.neuralnet.connections.Weight
import elaio.neuralnet.processing.GraphTraversal

object WeightInitializer {
  // what one initialization touched - the two numbers differ exactly where a cell is shared
  final case class Result(weightCells: Long, connections: Long)

  // He-style init, inverted because collectInConnections averages rather than sums
  def initialize(order: GraphTraversal.ReverseOrder): Result = {
    val random = new scala.util.Random
    // A shared cell belongs to several connections, so drawing per connection would keep nothing but
    // the last draw. Each cell is therefore drawn once - its sharers sit at the same position in
    // identically built containers and so agree on the fan-in the deviation is taken from.
    val drawnCells = mutable.HashSet.empty[Weight]
    var connections = 0L

    for (neuron <- order.sequence)
      if (neuron.connectionsIn.length > 0) {
        val deviation = math.sqrt(neuron.connectionsIn.length / Activation.secondMomentFactor)
        for (connectionIn <- neuron.connectionsIn) {
          connections = connections + 1L
          if (drawnCells.add(connectionIn.weightCell))
            connectionIn.weight = random.nextGaussian() * deviation
        }
      }

    Result(drawnCells.size.toLong, connections)
  }
}
