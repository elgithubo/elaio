package elaio.neuralnet.bigdata.container

import elaio.neuralnet.connections.Connection
//import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.Neuron
import elaio.neuralnet.units.NeuronType

class TensoredContainer(
    dimOuter: Int,
    inOutWidth: Int,
    dataCreator: DataCreator,
) {

  private var _inputNodes = Array.ofDim[Neuron](0)
  private var _outputNodes = Array.ofDim[Neuron](0)

  def inputNodes: Array[Neuron] = _inputNodes
  def outputNodes: Array[Neuron] = _outputNodes

  def init(): Array[Array[Neuron]] = {
    val result =
      buildRootNodes(
        dimOuter,
        inOutWidth,
        dataCreator
      )
    _inputNodes = result(0)
    _outputNodes = result(1)
    result
  }

  private def buildRootNodes(
      buildDimOuter: Int,
      buildInOutWidth: Int,
      dataCreator: DataCreator
  ): Array[Array[Neuron]] = {
    buildNodesRecurse(
      buildDimOuter,
      buildInOutWidth,
      dataCreator,
      true
    )
  }

  private def buildNodesRecurse(
      buildDimOuter: Int,
      buildInOutWidth: Int,
      dataCreator: DataCreator,
      inputBackpropagationCreationPossible: Boolean,
  ): Array[Array[Neuron]] = {
    var neuronsReturn = Array.ofDim[Neuron](3, 0)

    if (inputBackpropagationCreationPossible) {
        for (i <- 1 to buildInOutWidth)
          neuronsReturn(0) = neuronsReturn(0) :+ dataCreator.create(NeuronType.Input)
        for (i <- 1 to buildInOutWidth)
          neuronsReturn(1) = neuronsReturn(1) :+ dataCreator.create(NeuronType.Output)
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
        for (i <- 1 to buildInOutWidth) {
          var newNeuronSameRank = dataCreator.create(NeuronType.Hidden)
          newNeuronsSameRank = newNeuronsSameRank :+ newNeuronSameRank
          newNeuronsHere = newNeuronsHere :+ newNeuronSameRank
        }

        // determine if the node receives reverse wiring
        var isReverseNode: Boolean = true
        if (
          nextNeuronOuterIndexOffset > 0 && (buildDimOuter - nextNeuronOuterIndexOffset.abs) % 2 == 0 ||
          nextNeuronOuterIndexOffset < 0 && (buildDimOuter - nextNeuronOuterIndexOffset.abs) % 2 == 1
        ) isReverseNode = false

        if (!isReverseNode)
          if (nextNeuronOuterIndexOffset == buildDimOuter)
            for (inNeuron <- neuronsReturn(0))
              newNeuronsHere.foreach( connectNeurons(inNeuron, _) )

        if (buildDimOuter > 1) {
          var neuronsLowerDim = buildNodesRecurse(
            buildDimOuter - 1,
            buildInOutWidth + 1,
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
        /*if (isReverseNode)
          if (nextNeuronOuterIndexOffset == -buildDimOuter)
            for (outNeuron <- neuronsReturn(1))
              newNeuronsHere.foreach( connectNeurons(_, outNeuron))*/
        // outputs read every reverse group and its child level - widens the output cut
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
    val connection = new Connection {
      protected var _neuronSource: Neuron = connectionNeuronSource
      protected var _neuronTarget: Neuron = connectionNeuronTarget
    }
    connection.neuronTarget.addInConnection(connection)
    connection.neuronSource.addOutConnection(connection)
  }
}
