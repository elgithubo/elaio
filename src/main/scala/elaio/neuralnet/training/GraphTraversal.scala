package elaio.neuralnet.training

import scala.collection.mutable
import elaio.neuralnet.units.Neuron

object GraphTraversal {
  final case class ReverseOrder(
      sequence: Vector[Neuron],
      reachable: Set[Neuron],
      outputs: Set[Neuron]
  )

  private val reverseOrderCache = mutable.HashMap.empty[Set[Neuron], ReverseOrder]

  // Returns all neurons reachable from outputs in reverse-topological order:
  // sinks (usually outputs) first, then their sources towards inputs.
  def reverseTopologicalFromOutputs(outputNodes: Array[Neuron]): ReverseOrder = {
    val outputSet = outputNodes.toSet
    reverseOrderCache.getOrElseUpdate(outputSet, {
      val sequence = computeReverseTopologicalFromOutputs(outputSet)
      ReverseOrder(sequence, sequence.toSet, outputSet)
    })
  }

  private def computeReverseTopologicalFromOutputs(outputSet: Set[Neuron]): Vector[Neuron] = {
    val visited = mutable.Set.empty[Neuron]
    val postOrder = mutable.ArrayBuffer.empty[Neuron]

    for (start <- outputSet.toVector.sortBy(-_.id)) {
      val stack = mutable.Stack[(Neuron, Boolean)]((start, false))

      while (stack.nonEmpty) {
        val (neuron, expanded) = stack.pop()
        if (expanded) {
          postOrder += neuron
        } else if (visited.add(neuron)) {
          stack.push((neuron, true))
          for (source <- neuron.connectionsIn.iterator.map(_.neuronSource).toSet.toVector.sortBy(_.id).reverseIterator)
            stack.push((source, false))
        }
      }
    }

    postOrder.reverse.toVector
  }
}
