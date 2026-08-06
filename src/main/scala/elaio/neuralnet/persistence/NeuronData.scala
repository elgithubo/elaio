package elaio.neuralnet.persistence

// to be serialized part of a neuron
final case class NeuronData(
    id: Long,
    neuronType: Byte,
    bias: Double
)
