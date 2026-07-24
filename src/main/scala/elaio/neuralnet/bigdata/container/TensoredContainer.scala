package elaio.neuralnet.bigdata.container

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.Neuron
import elaio.neuralnet.units.NeuronType

class TensoredContainer(
    dimOuter: Int,
    inOutWidth: Int,
    dataCreator: DataCreator,
    recurse: Boolean,
) {

  class BuildData {}

  var _inputNodes = Array.ofDim[Neuron](0)
  var _outputNodes = Array.ofDim[Neuron](0)

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
      null,
      dataCreator,
      true
    )
  }

  private def buildNodesRecurse(
      buildDimOuter: Int,
      buildInOutWidth: Int,
      neurons: Array[Neuron],
      dataCreator: DataCreator,
      inputBackpropagationCreationPossible: Boolean,
  ): Array[Array[Neuron]] = {
    var neuronsReturn = Array.ofDim[Neuron](3, 0)
    var neuronsLastLayer = Array.ofDim[Neuron](0)

    if (inputBackpropagationCreationPossible) {
        for (i <- 1 to buildInOutWidth) {
          neuronsReturn(0) =
            neuronsReturn(0) :+ dataCreator.create(NeuronType.Input)
        }
        for (i <- 1 to buildInOutWidth) {
          neuronsReturn(1) =
            neuronsReturn(1) :+ dataCreator.create(NeuronType.Output)
        }
    }

    var bottomNeuronsLastRecur: Array[Neuron] = Array.ofDim[Neuron](0)
    var newNeuronsSameRank: Array[Neuron] = Array.ofDim[Neuron](0)
    var hereNeuronsLastToConnect: Array[Neuron] = Array.ofDim[Neuron](0)

    for (nextNeuronOuterIndexOffset <- buildDimOuter to -buildDimOuter by -1) {
      if (nextNeuronOuterIndexOffset != 0) {
        var newNeuronsHere: Array[Neuron] = Array.ofDim[Neuron](0)
        var bottomNeuronsThisRecur: Array[Neuron] = Array.ofDim[Neuron](0)
        for (i <- 1 to buildInOutWidth) {
          var newNeuronSameRank = dataCreator.create(NeuronType.Hidden)
          newNeuronsSameRank = newNeuronsSameRank :+ newNeuronSameRank
          newNeuronsHere = newNeuronsHere :+ newNeuronSameRank
        }

        //TODO this if doubles a lot of coding, should be simplified
        if (
          nextNeuronOuterIndexOffset > 0 && (buildDimOuter - nextNeuronOuterIndexOffset.abs) % 2 == 0 ||
          nextNeuronOuterIndexOffset < 0 && (buildDimOuter - nextNeuronOuterIndexOffset.abs) % 2 == 1
        ) {
          if (nextNeuronOuterIndexOffset == buildDimOuter) {
            for (inNeuron <- neuronsReturn(0)) {
              newNeuronsHere.foreach(
                connectNeurons(inNeuron, _)
              )
            }
          }
          if (buildDimOuter > 1) {
            var neuronsLowerDim = buildNodesRecurse(
              buildDimOuter - 1,
              1,
              newNeuronsHere,
              dataCreator,
              false,
            )
            neuronsReturn(2) = neuronsLowerDim(2)

            bottomNeuronsThisRecur = neuronsLowerDim(2)

            for (neuronLowerDim <- neuronsLowerDim(0)) {
              newNeuronsHere.foreach(
                connectNeurons(_, neuronLowerDim)
              )
            }
          } else {
            neuronsReturn(2) = neuronsReturn(2) ++ newNeuronsHere
          }
        } else {
          if (buildDimOuter > 1) {
            var neuronsLowerDim = buildNodesRecurse(
              buildDimOuter - 1,
              1,
              newNeuronsHere,
              dataCreator,
              false,
            )
            neuronsReturn(2) = neuronsLowerDim(2)

            bottomNeuronsThisRecur = neuronsLowerDim(2)

            for (neuronLowerDim <- neuronsLowerDim(0)) {
              newNeuronsHere.foreach(
                connectNeurons(neuronLowerDim, _)
              )
            }
          } else {
            neuronsReturn(2) = neuronsReturn(2) ++ newNeuronsHere
          }
          if (nextNeuronOuterIndexOffset == -buildDimOuter) {
            for (outNeuron <- neuronsReturn(1)) {
              newNeuronsHere.foreach(
                connectNeurons(_, outNeuron)
              )
            }
          }
        }
        if (buildDimOuter > 1) {
          if (bottomNeuronsLastRecur.length > 0) {
            for (bottomNeuronLastRecur <- bottomNeuronsLastRecur) {
              for (bottomNeuronThisRecur <- bottomNeuronsThisRecur) {
                connectNeurons(bottomNeuronLastRecur, bottomNeuronThisRecur)
              }
            }
          }
          bottomNeuronsLastRecur = bottomNeuronsThisRecur
        }
        if (hereNeuronsLastToConnect.length > 0) {
          for (hereNeuronLastToConnect <- hereNeuronsLastToConnect) {
            newNeuronsHere.foreach(
              connectNeurons(hereNeuronLastToConnect, _)
            )
          }
        }
        hereNeuronsLastToConnect = newNeuronsHere
      }
    }

    if (!inputBackpropagationCreationPossible) {
      neuronsReturn(0) = newNeuronsSameRank
    }
    neuronsReturn
  }

  private def connectNeurons(
      connectionNeuronSource: Neuron,
      connectionNeuronTarget: Neuron
  ): Unit = {
    //connectionNeuronTarget.init(0, connectionNeuronSource.target, connectionNeuronSource.tolerance)
    val connection = new Connection {
      override val neuronSource: Neuron = connectionNeuronSource
      override val neuronTarget: Neuron = connectionNeuronTarget
    }
    connection.getNeuronTarget.addInConnection(connection)
    connection.getNeuronSource.addOutConnection(connection)
  }
}
