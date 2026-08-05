package elaio.neuralnet.persistence

import scala.collection.mutable.HashMap
import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.connections.Connection
import elaio.neuralnet.processing.GraphTraversal
import elaio.neuralnet.units.{HiddenNeuronLeakyRelu, HiddenNeuronSquare, InputNeuron, Neuron, NeuronType, OutputNeuron}

object NetworkStateMapper {

  def capture(container: TensoredContainer): StateContainer = {
    val neurons = GraphTraversal.reverseTopologicalFromOutputs(container.outputNodes).sequence
    val connections = connectionsFrom(neurons)
    val neuronStore = HashMap.from(neurons.map { neuron =>
      require(java.lang.Double.isFinite(neuron.bias), s"Neuron ${neuron.id} has a non-finite bias")
      neuron.id -> NeuronData(
        neuron.id,
        (neuron match {
          case _: InputNeuron  => NeuronType.Input
          case _: HiddenNeuronLeakyRelu => NeuronType.HiddenLeakyRelu
          case _: HiddenNeuronSquare => NeuronType.HiddenSquare
          case _: OutputNeuron => NeuronType.Output
          case _ => throw new IllegalArgumentException(s"Unsupported neuron type ${neuron.getClass.getName}")
        }).id.toByte,
        neuron.bias
      )
    })
    val connectionStore = HashMap.from(connections.map { connection =>
      require(java.lang.Double.isFinite(connection.weight), s"Connection ${connection.id} has a non-finite weight")
      connection.id -> ConnectionData(
        connection.id,
        connection.neuronSource.id,
        connection.neuronTarget.id,
        connection.weight
      )
    })
    StateContainer(neuronStore, connectionStore)
  }

  def restore(stateContainer: StateContainer, container: TensoredContainer): Unit = {
    val neurons = GraphTraversal.reverseTopologicalFromOutputs(container.outputNodes).sequence
    val connections = connectionsFrom(neurons)
    val neuronsById = neurons.map(neuron => neuron.id -> neuron).toMap
    val connectionsById = connections.map(connection => connection.id -> connection).toMap

    for ((id, data) <- stateContainer.connectionStore) {
      require(data.id == id, s"Connection store key $id does not match contained ID ${data.id}") //you never know
      require(java.lang.Double.isFinite(data.weight), s"Persisted connection $id has a non-finite weight")
      val connection = connectionsById(id)
      require(
        data.sourceId == connection.neuronSource.id && data.targetId == connection.neuronTarget.id,
        s"Persisted connection $id has different endpoints than the current network"
      )
    }

    for ((id, data) <- stateContainer.neuronStore)
      neuronsById(id).bias = data.bias
    for ((id, data) <- stateContainer.connectionStore)
      connectionsById(id).weight = data.weight
  }

  private def connectionsFrom(neurons: Iterable[Neuron]): Vector[Connection] =
    neurons.iterator.flatMap(_.connectionsIn).toVector.distinct

}
