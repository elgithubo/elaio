package elaio.neuralnet.bigdata

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import scala.concurrent.ExecutionContext.Implicits.global
import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.connections.Connection
import elaio.neuralnet.processing.{GraphTraversal, NeuronCollectionCache}
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

  // the token inputs end to end - a convenience view; initInputs addresses the containers directly
  def inputNodes: Array[InputNeuron] = _inputNodes
  def outputNodes: Array[OutputNeuron] = _outputNodes
  def reverseOrder: GraphTraversal.ReverseOrder =
    if (_reverseOrder != null) _reverseOrder
    else throw new IllegalStateException("container has not been initialized")

  def tokenContainers: Vector[TensoredContainer] = containers

  // one cache per token container, each confined to the task that runs its container
  private val tokenCaches = containers.map(_ => new NeuronCollectionCache)

  // The token containers share no neurons and only read the shared weights during a forward pass,
  // so they run concurrently. The read-out then collects from a cache seeded with the finished
  // token outputs, so it never descends into the token graphs again.
  override def forward(cache: NeuronCollectionCache): Unit =
    if (containers.length == 1) super.forward(cache)
    else {
      val passes = containers.zip(tokenCaches).map { (container, tokenCache) =>
        Future {
          tokenCache.clear()
          for (outputNode <- container.outputNodes) outputNode.collectInConnections(tokenCache)
        }
      }
      passes.foreach(Await.result(_, Duration.Inf))
      cache.clear()
      for (container <- containers; tokenOutput <- container.outputNodes) cache.add(tokenOutput)
      for (outputNode <- _outputNodes) outputNode.collectInConnections(cache)
    }

  // each token goes to its own container, addressed directly rather than through the joined
  // inputNodes - that keeps the token layout an enforced contract instead of a shared assumption
  def initInputs(tokens: TokenMatrix): Unit = {
    require(tokens.length == containers.length, "expected " + containers.length + " tokens but got " + tokens.length)
    for (tokenIndex <- tokens.indices) {
      val expectedWidth = containers(tokenIndex).inputNodes.length
      require(tokens(tokenIndex).length == expectedWidth, "token " + tokenIndex + " needs " + expectedWidth + " values")
    }
    for (tokenIndex <- tokens.indices) {
      val values = tokens(tokenIndex)
      val nodes = containers(tokenIndex).inputNodes
      for (index <- values.indices) nodes(index).initInput(values(index))
    }
  }

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
      require(sharedConnections.length == referenceConnections.length, "stacked containers must have alike connections")
      require(sharedNeurons.length == referenceNeurons.length, "stacked containers must have alike neurons")
      for ((reference, shared) <- referenceConnections.zip(sharedConnections))
        shared.weightCell = reference.weightCell
      for ((reference, shared) <- referenceNeurons.zip(sharedNeurons))
        shared.biasCell = reference.biasCell
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
