package elaio.neuralnet.training

import scala.collection.mutable
import elaio.neuralnet.units.Neuron

object GraphTraversal {
  private val reverseOrderCache = mutable.HashMap.empty[Set[Neuron], Vector[Neuron]]

  def cacheSize: Int = reverseOrderCache.size

  // Returns all neurons reachable from outputs in reverse-topological order:
  // sinks (usually outputs) first, then their sources towards inputs.
  def reverseTopologicalFromOutputs(outputNodes: Array[Neuron]): Vector[Neuron] = {
    val cacheKey = outputNodes.toSet
    reverseOrderCache.getOrElseUpdate(cacheKey, computeReverseTopologicalFromOutputs(cacheKey))
  }

  private def computeReverseTopologicalFromOutputs(outputSet: Set[Neuron]): Vector[Neuron] = {
    val visited = mutable.Set.empty[Neuron]
    val postOrder = mutable.ArrayBuffer.empty[Neuron]

    for (start <- outputSet.toVector.sortBy(-_.id))
      if (visited.add(start)) {
        val stack = mutable.Stack[(Neuron, Boolean)]((start, false))

        while (stack.nonEmpty) {
          val (neuron, expanded) = stack.pop()
          if (expanded) {
            postOrder += neuron
          } else {
            stack.push((neuron, true))
            val sources = neuron.connectionsIn.iterator.map(_.getNeuronSource).toSet.toVector.sortBy(_.id)
            for (source <- sources.reverseIterator) if (visited.add(source)) stack.push((source, false))
          }
        }
      }

    postOrder.reverse.toVector
  }
}
