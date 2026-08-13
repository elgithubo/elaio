package elaio.neuralnet.bigdata

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.connections.Connection
import elaio.neuralnet.processing.{Backpropagation, GraphTraversal, NeuronCollectionCache}
import elaio.neuralnet.units.{InputNeuron, Neuron, OutputNeuron}

// the basis structure of a neural network, which is a directed graph of neurons and connections
trait NeuronNetwork(ids: IdAllocator) {
  // one allocator for the whole stack - the collection cache and the model files key on the neuron
  // id, so two neurons of one graph carrying the same id would be confused for each other

  def inputNodes: Array[InputNeuron]
  def outputNodes: Array[OutputNeuron]
  def reverseOrder: GraphTraversal.ReverseOrder

  // Feed one example. Only the network knows whether its input is one row or a matrix, so the
  // token rows are resolved here and nowhere else - a single container joins them, a stack hands
  // one row to each of its containers.
  def initInputs(tokens: TokenMatrix): Unit

  // one full forward pass - overridden where independent parts of the graph can run concurrently
  def forward(cache: NeuronCollectionCache): Unit = {
    cache.clear()
    for (outputNode <- outputNodes) outputNode.collectInConnections(cache)
  }

  // the delta phase of backpropagation - overridden where disjoint graph slices can run
  // concurrently; applying the updates stays serial because the parameters are shared
  def calculateDeltas(): Unit = {
    val order = reverseOrder
    Backpropagation.calculateOutputDeltas(order)
    Backpropagation.calculateDeltas(order.sequence, order.reachable, order.outputs)
  }

  protected final def connectNeurons(
      connectionNeuronSource: Neuron,
      connectionNeuronTarget: Neuron
  ): Unit = {
    val connection = new Connection(ids.nextConnectionId()) {
      protected var _neuronSource: Neuron = connectionNeuronSource
      protected var _neuronTarget: Neuron = connectionNeuronTarget
    }
    connection.neuronTarget.addInConnection(connection)
    connection.neuronSource.addOutConnection(connection)
  }
}
