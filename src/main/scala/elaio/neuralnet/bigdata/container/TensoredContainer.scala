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
  var _backpropagationNodes = Array.ofDim[Neuron](0)

  def inputNodes: Array[Neuron] = _inputNodes
  def backpropagationNodes: Array[Neuron] = _backpropagationNodes

  def init(): Array[Array[Neuron]] = {
    val result =
      buildRootNodes(
        dimOuter,
        inOutWidth,
        dataCreator,
        recurse,
      )
    _inputNodes = result(0)
    _backpropagationNodes = result(1)
    result
  }

  private def buildRootNodes(
      buildDimOuter: Int,
      buildInOutWidth: Int,
      dataCreator: DataCreator,
      buildRecurse: Boolean,
  ): Array[Array[Neuron]] = {
    if (buildRecurse)
      buildNodesRecurse(
        buildDimOuter,
        buildInOutWidth,
        null,
        dataCreator,
        true,
      )
    else
      buildNodesFlat(
        buildDimOuter,
        buildInOutWidth,
        null,
        dataCreator,
        true,
      )
  }

  private def buildNodesFlat(
      buildDimOuter: Int,
      buildInOutWidth: Int,
      neuron: Neuron,
      dataCreator: DataCreator,
      inputBackpropagationCreationPossible: Boolean,
  ): Array[Array[Neuron]] = {
    var neuronsInOutReturn = Array.ofDim[Neuron](2, 0)
    var neuronsLastLayer = Array.ofDim[Neuron](0)

    for (nextNeuronOuterIndex <- buildDimOuter to -buildDimOuter by -1) {
      if (nextNeuronOuterIndex != 0) {

        var neuronsCurrentLayer = Array.ofDim[Neuron](0)

        var loopCount: Int = 0
        if (
          inputBackpropagationCreationPossible && buildDimOuter - nextNeuronOuterIndex.abs == 0 && nextNeuronOuterIndex < 0
        ) {
          loopCount = 1
        } else {
          loopCount =
            buildInOutWidth * (buildDimOuter - nextNeuronOuterIndex.abs + 1)
        }
        for (index <- 1 to loopCount) {
          val newNeuronSameRank = dataCreator.create(
            if (
              inputBackpropagationCreationPossible && buildDimOuter - nextNeuronOuterIndex.abs == 0 && nextNeuronOuterIndex > 0
            )
              NeuronType.Input
            else if (
              inputBackpropagationCreationPossible && buildDimOuter - nextNeuronOuterIndex.abs == 0 && nextNeuronOuterIndex < 0
            )
              NeuronType.Backpropagation
            else
              NeuronType.Hidden
          )

          if (inputBackpropagationCreationPossible) {
            if (nextNeuronOuterIndex == buildDimOuter) {
              neuronsInOutReturn(0) = neuronsInOutReturn(0) :+ newNeuronSameRank
            }
          }

          neuronsCurrentLayer = neuronsCurrentLayer :+ newNeuronSameRank
        }

        for (neuronLastLayer <- neuronsLastLayer) {
          for (neuronCurrentLayer <- neuronsCurrentLayer) {
            connectNeurons(neuronLastLayer, neuronCurrentLayer)
          }
        }
        neuronsLastLayer = neuronsCurrentLayer
      }
    }

    neuronsInOutReturn
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
            neuronsReturn(1) :+ dataCreator.create(NeuronType.Backpropagation)
        }
    }

    var bottomNeuronsLastRecur: Array[Neuron] = Array.ofDim[Neuron](0)
    var newNeuronsSameRank: Array[Neuron] = Array.ofDim[Neuron](0)
    var topLoopCount: Int = 0
    var topNeuronsLastToConnect: Array[Neuron] = Array.ofDim[Neuron](0)

    for (nextNeuronOuterIndexOffset <- buildDimOuter to -buildDimOuter by -1) {
      if (nextNeuronOuterIndexOffset != 0) {
        topLoopCount = topLoopCount + 1

        var newNeuronsTop: Array[Neuron] = Array.ofDim[Neuron](0)
        var newNeuronsHere: Array[Neuron] = Array.ofDim[Neuron](0)
        var bottomNeuronsThisRecur: Array[Neuron] = Array.ofDim[Neuron](0)
        for (i <- 1 to buildInOutWidth) {
          var newNeuronSameRank = dataCreator.create(NeuronType.Hidden)
          newNeuronsSameRank = newNeuronsSameRank :+ newNeuronSameRank
          if (inputBackpropagationCreationPossible) {
            newNeuronsTop = newNeuronsTop :+ newNeuronSameRank
          }
          newNeuronsHere = newNeuronsHere :+ newNeuronSameRank
        }

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
            neuronsReturn(2) = neuronsReturn(2) ++ newNeuronsSameRank
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
            neuronsReturn(2) = neuronsReturn(2) ++ newNeuronsSameRank
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
            bottomNeuronsLastRecur = Array.ofDim[Neuron](0)
          } else {
            bottomNeuronsLastRecur = bottomNeuronsThisRecur
          }
        }
        if (inputBackpropagationCreationPossible) {
          if (topLoopCount % 2 == 0) {
            topNeuronsLastToConnect = newNeuronsTop
          } else if (topNeuronsLastToConnect != null) {
            for (topNeuronLastToConnect <- topNeuronsLastToConnect) {
              newNeuronsTop.foreach(
                connectNeurons(topNeuronLastToConnect, _)
              )
            }
            newNeuronsTop = null
          }
        }
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
    val connection = new Connection {
      override val neuronSource: Neuron = connectionNeuronSource
      override val neuronTarget: Neuron = connectionNeuronTarget
    }
    connection.getNeuronTarget.addInConnection(connection)

  }
}
