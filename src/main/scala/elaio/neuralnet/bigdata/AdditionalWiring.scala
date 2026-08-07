package elaio.neuralnet.bigdata

import elaio.neuralnet.units.Neuron

final case class NeuronGroup(depth: Int, neurons: Vector[Neuron])

trait AdditionalWiring {
  def wire(context: AdditionalWiring.Context): Unit
}

object AdditionalWiring {
  final class Context private[bigdata] (
      val groups: Vector[NeuronGroup],
      connectNeurons: (Neuron, Neuron) => Unit
  ) {
    private val depthByNeuron =
      groups.iterator.flatMap(group => group.neurons.map(_ -> group.depth)).toMap

    def connect(source: Neuron, target: Neuron): Unit = {
      require(
        depthByNeuron(source) < depthByNeuron(target),
        "additional wiring must connect an earlier group to a later group"
      )
      connectNeurons(source, target)
    }
  }
}
