package elaio.neuralnet.persistence

import scala.collection.mutable.HashMap

@SerialVersionUID(1L)
final case class StateContainer(
    neuronStore: HashMap[Long, NeuronData],
    connectionStore: HashMap[Long, ConnectionData]
) extends Serializable
