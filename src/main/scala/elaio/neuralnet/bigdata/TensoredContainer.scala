package elaio.neuralnet.bigdata

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.processing.GraphTraversal
//import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{HiddenNeuron, Neuron, NeuronDataCreator, NeuronType, InputNeuron, OutputNeuron, IntermediateOutputNeuron}
import elaio.neuralnet.bigdata.interface.{TensoredContainerInternal, TensoredContainerInOut}

// represents a multi-dimensional tensor of dimension dimOuter
class TensoredContainer(
    dimOuter: Int,
    inWidth: Int,
    outWidth: Int,
    dataCreator: NeuronDataCreator,
    additionalWiring: Option[AdditionalWiring] = None,
    // a stack of containers passes one allocator to all of them, so their ids stay distinct
    ids: IdAllocator = new IdAllocator,
    intermediateOutputs: Boolean = false
) extends NeuronNetwork(ids) {

  private val _intermediateOutputs: Boolean = intermediateOutputs

  private var _inputNodes = Array.ofDim[InputNeuron](0)
  private var _outputNodes = Array.ofDim[OutputNeuron](0)
  private var _reverseOrder: GraphTraversal.ReverseOrder = null

  def inputNodes: Array[InputNeuron] = _inputNodes
  def outputNodes: Array[OutputNeuron] = _outputNodes
  def reverseOrder: GraphTraversal.ReverseOrder =
    if( _reverseOrder != null) _reverseOrder else throw new IllegalStateException("container has not been initialized")

  // One flat input row, so the token rows are read end to end. Only the total is checked, not how
  // the values are split into rows - a single container has no token structure to violate. That
  // makes it laxer than LayeredContainer on purpose: a wrong token width fails there and passes
  // here, so a shape error only surfaces once the same test runs on a stack.
  def initInputs(tokens: TokenMatrix): Unit = {
    require(
      tokens.iterator.map(_.length).sum == _inputNodes.length,
      "expected " + _inputNodes.length + " input token values but got " + tokens.iterator.map(_.length).sum
    )
    var index = 0
    for (token <- tokens; value <- token) {
      _inputNodes(index).initInput(value)
      index += 1
    }
  }

  def init(): Unit = {
    val result = buildRootNodes(
        dimOuter,
        inWidth,
        outWidth,
        dataCreator
      )
    _inputNodes = result.inputNodes
    _outputNodes = result.outputNodes
    val baseOrder = GraphTraversal.reverseTopologicalFromOutputs(_outputNodes)
    _reverseOrder = additionalWiring match {
      case Some(wiring) =>
        val context =
          new AdditionalWiring.Context(GraphTraversal.depthGroups(baseOrder), connectNeuronsIfMissing)
        wiring.wire(context)
        GraphTraversal.reverseTopologicalFromOutputs(_outputNodes)
      case None => baseOrder
    }
  }

  private def buildRootNodes(
      buildDimOuter: Int,
      buildInWidth: Int,
      buildOutWidth: Int,
      dataCreator: NeuronDataCreator
  ): TensoredContainerInOut = {
    val receivedResult = buildNodesRecurse(
      buildDimOuter,
      buildInWidth,
      buildOutWidth,
      dataCreator,
      true
    )
    val result = new TensoredContainerInOut
    result.inputNodes = receivedResult.inputNodes.map(_.asInstanceOf[InputNeuron])
    result.outputNodes = receivedResult.outputNodes.map(_.asInstanceOf[OutputNeuron])
    result
  }

  private def buildNodesRecurse(
      buildDimOuter: Int,
      buildInWidth: Int,
      buildOutWidth: Int,
      dataCreator: NeuronDataCreator,
      inputBackpropagationCreationPossible: Boolean
  ): TensoredContainerInternal = {
    var neuronsReturn = new TensoredContainerInternal

    if (inputBackpropagationCreationPossible) {
        for (i <- 1 to buildInWidth)
          neuronsReturn.addInputNode(dataCreator.create(NeuronType.Input, ids.nextNeuronId()).asInstanceOf[InputNeuron])
        for (i <- 1 to buildOutWidth)
          neuronsReturn.addOutputNode(
            if (!_intermediateOutputs)
              dataCreator.create(NeuronType.Output, ids.nextNeuronId()).asInstanceOf[OutputNeuron]
            else
              dataCreator.create(NeuronType.IntermediateOutput, ids.nextNeuronId()).asInstanceOf[IntermediateOutputNeuron]
          )
    }

    var bottomNeuronsLastRecur: Array[HiddenNeuron] = Array.ofDim[HiddenNeuron](0)
    var newNeuronsSameRank: Array[Neuron] = Array.ofDim[Neuron](0)
    var hereNeuronsLastToConnect: Array[HiddenNeuron] = Array.ofDim[HiddenNeuron](0)
    var childNeuronsLastRecur: Array[Neuron] = Array.ofDim[Neuron](0)

    for (nextNeuronOuterIndexOffset <- buildDimOuter to -buildDimOuter by -1) {
      if (nextNeuronOuterIndexOffset != 0) {
        var newNeuronsHere: Array[HiddenNeuron] = Array.ofDim[HiddenNeuron](0)
        var bottomNeuronsThisRecur: Array[HiddenNeuron] = Array.ofDim[HiddenNeuron](0)
        var childNeuronsThisRecur: Array[Neuron] = Array.ofDim[Neuron](0)
        var lowerDimNeuronsThisRecur: Array[Neuron] = Array.ofDim[Neuron](0)

        // determine if the node receives reverse wiring
        var isReverseNode: Boolean = true
        if (
          nextNeuronOuterIndexOffset > 0 && (buildDimOuter - nextNeuronOuterIndexOffset.abs) % 2 == 0 ||
          nextNeuronOuterIndexOffset < 0 && (buildDimOuter - nextNeuronOuterIndexOffset.abs) % 2 == 1
        ) isReverseNode = false

        for (i <- 1 to buildInWidth.max(buildOutWidth)) {
          var newNeuronSameRank =
            dataCreator.create(
              /* old implementation
              if (
                inputBackpropagationCreationPossible &&
                  nextNeuronOuterIndexOffset == buildDimOuter && i <= inWidth
                  //nextNeuronOuterIndexOffset % 2 == 0 && i <= inWidth => would double the number of square neurons
              ) NeuronType.HiddenSquare else NeuronType.HiddenLeakyRelu,*/
              if (inputBackpropagationCreationPossible && !isReverseNode)
                NeuronType.HiddenSquare else NeuronType.HiddenLeakyRelu,
              ids.nextNeuronId()
            ).asInstanceOf[HiddenNeuron]
          newNeuronsSameRank = newNeuronsSameRank :+ newNeuronSameRank
          newNeuronsHere = newNeuronsHere :+ newNeuronSameRank
        }

        // keep this for safety purposes concerning future edits
        if (!isReverseNode)
          if (nextNeuronOuterIndexOffset == buildDimOuter)
            for (inNeuron <- neuronsReturn.inputNodes)
              newNeuronsHere.foreach( connectNeurons(inNeuron, _) )

        if (buildDimOuter > 1) {
          var neuronsLowerDim = buildNodesRecurse(
            buildDimOuter - 1,
            buildInWidth + 1,
            buildOutWidth + 1,
            dataCreator,
            false,
          )
          neuronsReturn.intermediateNodes = neuronsLowerDim.intermediateNodes
          lowerDimNeuronsThisRecur = neuronsLowerDim.inputNodes

          bottomNeuronsThisRecur = neuronsLowerDim.intermediateNodes
          if(buildDimOuter > 2) // avoid double connections
            childNeuronsThisRecur = neuronsLowerDim.inputNodes

          for (neuronLowerDim <- neuronsLowerDim.inputNodes)
            for (newNeuronHere <- newNeuronsHere)
              if (isReverseNode) connectNeurons(neuronLowerDim, newNeuronHere)
              else connectNeurons(newNeuronHere, neuronLowerDim)

          // add wiring for dead end neurons
          if (childNeuronsLastRecur.length > 0) {
            var childNeuronIndex: Int = 0
            for (childNeuron <- childNeuronsThisRecur) {
              if (childNeuron.connectionsIn.isEmpty) {
                connectNeurons(childNeuronsLastRecur(childNeuronIndex), childNeuron)
                //NetTrace.WriteMessage("fix1")
              }
              childNeuronIndex = childNeuronIndex + 1
            }
          }
          if (childNeuronsThisRecur.length > 0) {
            var childNeuronIndex: Int = 0
            for (childNeuron <- childNeuronsLastRecur) {
              if (childNeuron.connectionsOut.isEmpty) {
                connectNeurons(childNeuron, childNeuronsThisRecur(childNeuronIndex))
                //NetTrace.WriteMessage("fix2")
              }
              childNeuronIndex = childNeuronIndex + 1
            }
          }
          childNeuronsLastRecur = childNeuronsThisRecur
        } else {
          neuronsReturn.addIntermediateNodes(newNeuronsHere)
        }

        // connect the input layer to each forward group and its child rank.
        if (!isReverseNode)
          for (inNeuron <- neuronsReturn.inputNodes) {
            for (newNeuronHere <- newNeuronsHere) {
              if(!inNeuron.connectionsOut.exists(connection => connection.neuronTarget == newNeuronHere))  // avoid double connections
                connectNeurons(inNeuron, newNeuronHere)
              //else
                //NetTrace.WriteMessage("fix3")
            }
            lowerDimNeuronsThisRecur.foreach( connectNeurons(inNeuron, _) )
          }

        // connect each reverse group and its child rank directly to the output layer.
        if (isReverseNode)
          for (outNeuron <- neuronsReturn.outputNodes) {
            newNeuronsHere.foreach( connectNeurons(_, outNeuron))
            lowerDimNeuronsThisRecur.foreach( connectNeurons(_, outNeuron))
          }

        if (buildDimOuter > 1) {
          if (bottomNeuronsLastRecur.length > 0)
            for (bottomNeuronLastRecur <- bottomNeuronsLastRecur)
              for (bottomNeuronThisRecur <- bottomNeuronsThisRecur)
                connectNeurons(bottomNeuronLastRecur, bottomNeuronThisRecur)
          bottomNeuronsLastRecur = bottomNeuronsThisRecur
        }
        if (hereNeuronsLastToConnect.length > 0)
          for (hereNeuronLastToConnect <- hereNeuronsLastToConnect)
            newNeuronsHere.foreach( connectNeurons(hereNeuronLastToConnect, _))
        hereNeuronsLastToConnect = newNeuronsHere
      }
    }

    if (!inputBackpropagationCreationPossible)
      neuronsReturn.inputNodes = newNeuronsSameRank

    neuronsReturn
  }

  // needed to avoid double connections if an additional wiring provider is injected
  private def connectNeuronsIfMissing(
      connectionNeuronSource: Neuron,
      connectionNeuronTarget: Neuron
  ): Unit =
    if (!connectionNeuronSource.connectionsOut.exists(_.neuronTarget == connectionNeuronTarget))
      connectNeurons(connectionNeuronSource, connectionNeuronTarget)

}
