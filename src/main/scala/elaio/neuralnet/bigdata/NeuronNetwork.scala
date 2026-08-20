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

  // One full forward pass - overridden where independent parts of the graph can run concurrently.
  //
  // INVARIANT for every concurrent override, of forward and of calculateDeltas alike: the tasks are
  // submitted and then all awaited before the caller returns. Nothing here is volatile or
  // synchronized - the visibility of every value, delta, weight and bias between the calling thread
  // and the workers rests entirely on the happens-before edges that handing a task to an
  // ExecutionContext and awaiting its result provide. Dropping an await, letting a phase overlap
  // with the next one, or running two examples at once therefore breaks the memory model silently:
  // no compiler warning, no exception, just occasional wrong numbers. Verified bit-exact against a
  // serial run over 1200 passes - that verification is only valid while the pattern holds.
  def forward(cache: NeuronCollectionCache): Unit = {
    cache.clear()
    for (outputNode <- outputNodes) outputNode.collectInConnections(cache)
  }

  // The delta phase of backpropagation - overridden where disjoint graph slices can run
  // concurrently. Applying the updates stays serial on purpose: the parameters are shared between
  // the slices, so concurrent accumulation would race. See the invariant on forward.
  def calculateDeltas(): Unit = {
    val order = reverseOrder
    Backpropagation.seedDeltas(order)
    Backpropagation.calculateDeltas(order.sequence, order.outputs)
  }

  // Propagates direct deltas that were seeded on internal neurons without replacing them.
  def propagateSeededDeltas(): Unit =
    Backpropagation.propagateSeededDeltas(reverseOrder.sequence)

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
