package elaio.neuralnet.bigdata

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.processing.GraphTraversal
import elaio.neuralnet.units.{InputNeuron, Neuron, NeuronDataCreator, NeuronType, OutputNeuron}

// The token matrix, materialised. One container per row, all built alike and all sharing their
// weights, so the stack is one function applied to every token rather than N separate functions.
// It takes N containers rather than one pass per token because an elaio neuron holds a single
// value - N copies of the state is what lets every token be alive at the same time, which is what
// attention across tokens needs.
//
// The containers touch nowhere in their hidden ranks. Their only meeting point is a read-out layer
// every token feeds; picking which token matters is the attention's job, not the wiring's.
final class LayeredContainer(
    tokenCount: Int,
    dimOuter: Int,
    tokenWidth: Int,
    // how wide one token's representation is where the read-out picks it up
    tokenOutWidth: Int,
    outWidth: Int,
    dataCreator: NeuronDataCreator,
    additionalWiring: Option[AdditionalWiring] = None,
) extends NeuronNetwork {

  require(tokenCount > 0, "a layered container needs at least one token")

  // one allocator for the whole stack - the collection cache and the model files key on the neuron
  // id, so two neurons of one graph carrying the same id would be confused for each other
  private val ids = new IdAllocator
  private val containers = Vector.fill(tokenCount)(
    new TensoredContainer(dimOuter, tokenWidth, tokenOutWidth, dataCreator, additionalWiring, ids)
  )

  private var _inputNodes = Array.ofDim[InputNeuron](0)
  private var _outputNodes = Array.ofDim[OutputNeuron](0)
  private var _reverseOrder: GraphTraversal.ReverseOrder = null

  // the token inputs end to end, so a flattened token matrix lands in the right places
  def inputNodes: Array[InputNeuron] = _inputNodes
  def outputNodes: Array[OutputNeuron] = _outputNodes
  def reverseOrder: GraphTraversal.ReverseOrder =
    if (_reverseOrder != null) _reverseOrder
    else throw new IllegalStateException("container has not been initialized")

  def tokenContainers: Vector[TensoredContainer] = containers

  def init(): Unit = {
    containers.foreach(_.init())
    shareParameters()
    _inputNodes = containers.toArray.flatMap(_.inputNodes)
    _outputNodes = Array.fill(outWidth)(
      dataCreator.create(NeuronType.Output, ids.nextNeuronId()).asInstanceOf[OutputNeuron]
    )
    // a token container's outputs become hidden nodes of the stack: linear, and no target is set
    for {
      container <- containers
      tokenOutput <- container.outputNodes
      readOut <- _outputNodes
    } connectNeurons(tokenOutput, readOut)
    _reverseOrder = GraphTraversal.reverseTopologicalFromOutputs(_outputNodes)
  }

  // Every container is built by the same deterministic recursion, so its n-th connection and its
  // n-th neuron play the same role in every one of them. Pointing them all at the first container's
  // cells is therefore enough - no separate mapping is needed.
  //
  // Biases have to travel with the weights. Sharing only the weights would leave every token with
  // its own offset, so the stack would compute f_i(x) = activation(mean(W*x) + b_i) - one linear
  // part, N different functions. The read-out is left out on purpose: it is one layer, not one
  // per token.
  private def shareParameters(): Unit = {
    val referenceConnections = connectionsOf(containers.head)
    val referenceNeurons = neuronsOf(containers.head)
    for (container <- containers.tail) {
      val sharedConnections = connectionsOf(container)
      val sharedNeurons = neuronsOf(container)
      require(sharedConnections.length == referenceConnections.length, "stacked containers must be built alike")
      require(sharedNeurons.length == referenceNeurons.length, "stacked containers must be built alike")
      for ((reference, shared) <- referenceConnections.zip(sharedConnections))
        shared.weightParameter = reference.weightParameter
      for ((reference, shared) <- referenceNeurons.zip(sharedNeurons))
        shared.biasParameter = reference.biasParameter
    }
  }

  private def connectionsOf(container: TensoredContainer): Vector[Connection] =
    container.reverseOrder.sequence.flatMap(_.connectionsIn).sortBy(_.id)

  private def neuronsOf(container: TensoredContainer): Vector[Neuron] =
    container.reverseOrder.sequence.sortBy(_.id)

  private def connectNeurons(
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
