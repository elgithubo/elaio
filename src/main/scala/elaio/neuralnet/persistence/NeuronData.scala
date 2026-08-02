package elaio.neuralnet.persistence

final case class NeuronData(
    id: Long,
    neuronType: Byte,
    bias: Double
)
