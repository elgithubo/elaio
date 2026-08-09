package elaio.neuralnet.processing

import scala.collection.mutable
import elaio.neuralnet.connections.Weight
import elaio.neuralnet.units.Neuron

// one layer of the graph: every neuron whose longest path from a source has the same length
final case class NeuronGroup(depth: Int, neurons: Vector[Neuron])

object GraphTraversal {
  final case class ReverseOrder(
      sequence: Vector[Neuron],
      reachable: Set[Neuron],
      outputs: Set[Neuron],
      connectionWeights: Vector[Weight],
      hasSharedConnectionWeights: Boolean
  )

  // Layers the reachable neurons by their longest path from a source. Every edge runs from
  // a lower depth to a higher one, so the result is a valid topological layering.
  def depthGroups(order: ReverseOrder): Vector[NeuronGroup] = {
    val depthByNeuron = mutable.HashMap.empty[Neuron, Int]

    // order.sequence is reverse topological, so reading it backwards visits every
    // source before the neuron that reads it - one pass, no recursion, no stack limit
    for (neuron <- order.sequence.reverseIterator)
      depthByNeuron(neuron) =
        neuron.connectionsIn.iterator
          .map(_.neuronSource)
          .filter(order.reachable)
          .map(depthByNeuron)
          .maxOption
          .fold(0)(_ + 1)

    order.sequence
      .groupBy(depthByNeuron)
      .toVector
      .sortBy(_._1)
      .map { case (groupDepth, neurons) => NeuronGroup(groupDepth, neurons.sortBy(_.id)) }
  }

  // Returns all neurons reachable from outputs in reverse-topological order:
  // outputs first, then their sources towards inputs.
  // takes any array of neurons - scala arrays are invariant, so Array[OutputNeuron] needs the bound
  def reverseTopologicalFromOutputs(outputNodes: Array[? <: Neuron]): ReverseOrder = {
    val outputSet: Set[Neuron] = outputNodes.toSet
    val sequence = computeReverseTopologicalFromOutputs(outputSet)
    val allWeights = sequence.iterator.flatMap(_.connectionsIn).map(_.weightCell).toVector
    val connectionWeights = allWeights.distinct
    ReverseOrder(
      sequence,
      sequence.toSet,
      outputSet,
      connectionWeights,
      connectionWeights.length != allWeights.length
    )
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
        for (source <- neuron.connectionsIn.iterator.map(_.neuronSource).toSet.toVector.sortBy(-_.id))
          stack.push((source, false))
      }
    }

    neuronsReverseOrder.reverse.toVector
  }
}
