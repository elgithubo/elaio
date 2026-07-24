package elaio.neuralnet.training

import elaio.neuralnet.activation.Activation
import elaio.neuralnet.units.Neuron

object WeightInitializer {
  // He-style initialisation, adapted to the fact that Neuron.collectInConnections
  // averages its incoming values instead of summing them:
  //     z = (1 / N) * sum_i (w_i * a_i)   =>   Var(z) = Var(w) * E[a^2] / N
  // With E[a^2] = secondMomentFactor * Var(z_prev), holding the signal variance
  // steady from one layer to the next needs
  //     Var(w) = N / secondMomentFactor          (roughly 2N for leaky ReLU)
  //
  // Note this is the inverse of textbook He init, where Var(w) = 2/N. Dividing
  // the sum by N shrinks it faster (1/N) than N random terms accumulate
  // (sqrt(N)), so here the weights have to GROW with fan-in to compensate,
  // rather than shrink.
  //
  // Has to run after the graph is fully built: a Connection cannot size itself
  // at construction time, because its target keeps gaining further
  // in-connections afterwards and the fan-in is not final yet. Walking backward
  // from the outputs covers exactly the connections a forward pass will use.
  def initialize(outputNodes: Array[Neuron]): Int = {
    val random = new scala.util.Random
    val seen = scala.collection.mutable.Set[Double]()
    val pending = scala.collection.mutable.Stack[Neuron]()
    var connectionsInitialized = 0

    outputNodes.foreach(neuron => if (seen.add(neuron.id)) pending.push(neuron))

    while (pending.nonEmpty) {
      val neuron = pending.pop()
      val fanIn = neuron.connectionsIn.length
      if (fanIn > 0) {
        val deviation = math.sqrt(fanIn / Activation.secondMomentFactor)
        for (connectionIn <- neuron.connectionsIn) {
          connectionIn.weight = random.nextGaussian() * deviation
          connectionsInitialized = connectionsInitialized + 1
          val source = connectionIn.getNeuronSource
          if (seen.add(source.id)) pending.push(source)
        }
      }
    }
    connectionsInitialized
  }
}
