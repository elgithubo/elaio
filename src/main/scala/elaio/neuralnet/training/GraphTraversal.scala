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

  // correct only for an acyclic graph - recurrent connections would break the order silently
  private def computeReverseTopologicalFromOutputs(outputSet: Set[Neuron]): Vector[Neuron] = {
    val neuronsVisited = mutable.Set.empty[Neuron]
    val neuronsReverseOrder = mutable.ArrayBuffer.empty[Neuron]
    val stack = mutable.Stack.empty[(Neuron, Boolean)]

    for (start <- outputSet.toVector.sortBy(-_.id))
      stack.push((start, false))
    
    while (stack.nonEmpty) {
      val (neuron, expanded) = stack.pop()
      if (expanded) {
        neuronsReverseOrder += neuron
      } else if (neuronsVisited.add(neuron)) {
        stack.push((neuron, true))
        for (source <- neuron.connectionsIn.iterator.map(_.neuronSource).toSet.toVector.sortBy(-_.id).reverseIterator)
          stack.push((source, false))
      }
    }

    neuronsReverseOrder.reverse.toVector
  }
}
