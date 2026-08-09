package elaio.neuralnet.bigdata

// Hands out ids for one graph. A stack of containers must share one, because the collection cache
// and the model files key on the neuron id - two neurons in one graph may never carry the same.
final class IdAllocator {
  private var neuronIdCounter = 0L
  private var connectionIdCounter = 0L

  def nextNeuronId(): Long = {
    neuronIdCounter = neuronIdCounter + 1L
    neuronIdCounter
  }

  def nextConnectionId(): Long = {
    connectionIdCounter = connectionIdCounter + 1L
    connectionIdCounter
  }
}
