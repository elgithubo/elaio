package elaio.neuralnet.bigdata

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.processing.GraphTraversal
//import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.Neuron
import elaio.neuralnet.units.NeuronDataCreator
import elaio.neuralnet.units.NeuronType

class TensoredContainer(
    dimOuter: Int,
    inWidth: Int,
    outWidth: Int,
    dataCreator: NeuronDataCreator,
) {

  private var _inputNodes = Array.ofDim[Neuron](0)
  private var _outputNodes = Array.ofDim[Neuron](0)
  private var _reverseOrder: GraphTraversal.ReverseOrder = null
  private var neuronIdCounter = 0L
  private var connectionIdCounter = 0L

  def inputNodes: Array[Neuron] = _inputNodes
  def outputNodes: Array[Neuron] = _outputNodes
  def reverseOrder: GraphTraversal.ReverseOrder =
    if( _reverseOrder != null) _reverseOrder else throw new IllegalStateException("container has not been initialized")

  def init(): Array[Array[Neuron]] = {
    neuronIdCounter = 0L
    connectionIdCounter = 0L
    val result =
      buildRootNodes(
        dimOuter,
        inWidth,
        outWidth,
        dataCreator
      )
    _inputNodes = result(0)
    _outputNodes = result(1)
    _reverseOrder = GraphTraversal.reverseTopologicalFromOutputs(_outputNodes)
    result
  }

  private def buildRootNodes(
      buildDimOuter: Int,
      buildInWidth: Int,
      buildOutWidth: Int,
      dataCreator: NeuronDataCreator
  ): Array[Array[Neuron]] = {
    buildNodesRecurse(
      buildDimOuter,
      buildInWidth,
      buildOutWidth,
      dataCreator,
      true
    )
  }

  private def buildNodesRecurse(
      buildDimOuter: Int,
      buildInWidth: Int,
      buildOutWidth: Int,
      dataCreator: NeuronDataCreator,
      inputBackpropagationCreationPossible: Boolean,
  ): Array[Array[Neuron]] = {
    var neuronsReturn = Array.ofDim[Neuron](3, 0)

    if (inputBackpropagationCreationPossible) {
        for (i <- 1 to buildInWidth)
          neuronsReturn(0) = neuronsReturn(0) :+ dataCreator.create(NeuronType.Input, nextNeuronId())
        for (i <- 1 to buildOutWidth)
          neuronsReturn(1) = neuronsReturn(1) :+ dataCreator.create(NeuronType.Output, nextNeuronId())
    }

    var bottomNeuronsLastRecur: Array[Neuron] = Array.ofDim[Neuron](0)
    var newNeuronsSameRank: Array[Neuron] = Array.ofDim[Neuron](0)
    var hereNeuronsLastToConnect: Array[Neuron] = Array.ofDim[Neuron](0)
    var childNeuronsLastRecur: Array[Neuron] = Array.ofDim[Neuron](0)

    for (nextNeuronOuterIndexOffset <- buildDimOuter to -buildDimOuter by -1) {
      if (nextNeuronOuterIndexOffset != 0) {
        var newNeuronsHere: Array[Neuron] = Array.ofDim[Neuron](0)
        var bottomNeuronsThisRecur: Array[Neuron] = Array.ofDim[Neuron](0)
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
              nextNeuronId()
            )
          newNeuronsSameRank = newNeuronsSameRank :+ newNeuronSameRank
          newNeuronsHere = newNeuronsHere :+ newNeuronSameRank
        }

        // keep this for safety purposes concerning future edits
        if (!isReverseNode)
          if (nextNeuronOuterIndexOffset == buildDimOuter)
            for (inNeuron <- neuronsReturn(0))
              newNeuronsHere.foreach( connectNeurons(inNeuron, _) )

        if (buildDimOuter > 1) {
          var neuronsLowerDim = buildNodesRecurse(
            buildDimOuter - 1,
            buildOutWidth + 1,
            buildInWidth + 1,
            dataCreator,
            false,
          )
          neuronsReturn(2) = neuronsLowerDim(2)
          lowerDimNeuronsThisRecur = neuronsLowerDim(0)

          bottomNeuronsThisRecur = neuronsLowerDim(2)
          if(buildDimOuter > 2) // avoid double connections
            childNeuronsThisRecur = neuronsLowerDim(0)

          for (neuronLowerDim <- neuronsLowerDim(0))
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
          neuronsReturn(2) = neuronsReturn(2) ++ newNeuronsHere
        }

        // connect the input layer to each forward group and its child rank.
        if (!isReverseNode)
          for (inNeuron <- neuronsReturn(0)) {
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
          for (outNeuron <- neuronsReturn(1)) {
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
      neuronsReturn(0) = newNeuronsSameRank

    neuronsReturn
  }

  private def connectNeurons(
      connectionNeuronSource: Neuron,
      connectionNeuronTarget: Neuron
  ): Unit = {
    val connection = new Connection(nextConnectionId()) {
      protected var _neuronSource: Neuron = connectionNeuronSource
      protected var _neuronTarget: Neuron = connectionNeuronTarget
    }
    connection.neuronTarget.addInConnection(connection)
    connection.neuronSource.addOutConnection(connection)
  }

  private def nextNeuronId(): Long = {
    neuronIdCounter = neuronIdCounter + 1L
    neuronIdCounter
  }

  private def nextConnectionId(): Long = {
    connectionIdCounter = connectionIdCounter + 1L
    connectionIdCounter
  }
}
