package elaio.neuralnet.processing

import scala.collection.mutable
import elaio.neuralnet.connections.{Weight}
import elaio.neuralnet.units.Bias
import elaio.neuralnet.units.Neuron

// one layer of the graph: every neuron whose longest path from a source has the same length
final case class NeuronGroup(depth: Int, neurons: Vector[Neuron])

object GraphTraversal {
  // Snapshot of a fully wired graph; connections must not change after construction.
  final case class ReverseOrder(
      sequence: Vector[Neuron],
      reachable: Set[Neuron],
      outputs: Set[Neuron],
      connectionWeights: Vector[Weight],
      neuronBiases: Vector[Bias],
      hasSharedParameters: Boolean
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
    val allBiases = sequence.iterator.filter(_.connectionsIn.nonEmpty).map(_.biasCell).toVector
    val connectionWeights = allWeights.distinct
    val neuronBiases = allBiases.distinct
    val reachable = sequence.toSet
    requireEveryTargetCovered(sequence, reachable)

    ReverseOrder(
      sequence,
      reachable,
      outputSet,
      connectionWeights,
      neuronBiases,
      connectionWeights.length != allWeights.length || neuronBiases.length != allBiases.length
    )
  }

  // The delta sweep sums over every outgoing connection without asking whether the target belongs
  // to this order, which is only sound while every target does. That holds for every graph elaio
  // builds - the dead end repair in TensoredContainer sees to it - so this is checked once at
  // construction rather than per connection and per example. A topology that legitimately leaves a
  // target uncovered has to zero those deltas before the sweep instead; the check is here so that
  // change is a build failure and not a silently wrong gradient.
  private def requireEveryTargetCovered(sequence: Vector[Neuron], reachable: Set[Neuron]): Unit = {
    val uncovered = sequence.iterator.flatMap(_.connectionsOut).map(_.neuronTarget).filterNot(reachable)
    require(
      uncovered.isEmpty,
      "the graph feeds neurons this traversal does not cover - see requireEveryTargetCovered"
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
