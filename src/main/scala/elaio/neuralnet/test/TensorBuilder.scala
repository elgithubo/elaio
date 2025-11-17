package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{InputNeuron, NeuronDataCreator, OutputNeuron}
import elaio.neuralnet.processing.NeuronCollectionCache

object TensorBuilder {
  def run(): Unit = {
    // Enable the following line to write detailed trace messages to stdout
    NetTrace.started_(true)

    NetTrace.WriteMessage("start of test run")

    NetTrace.WriteMessage("backpropagation only working in pre-alpha mode")

    val neuronDataCreatorTensored = new NeuronDataCreator

    val container = new TensoredContainer(
      1,
      2,
      neuronDataCreatorTensored,
      true,
    )
    container.init()

    //val outValues: Array[Double] = feedbackIn(container, Array(6d, 5d, 4d, 3d, 2d, 1d), 0.5d, true)
    val outValues: Array[Double] = feedbackIn(container, Array(6d, 5d), 0.5d, true)
    for (outValue <- outValues) {
      NetTrace.WriteMessage("outValue: " + outValue)
    }

    NetTrace.WriteMessage("end of test run")
  }

  private def feedbackIn(container: TensoredContainer, inputValues: Array[Double], tolerance: Double, init: Boolean): Array[Double] = {
    var outValues: Array[Double] = Array.ofDim[Double](0)
    var outValuesCollected: Array[Double] = Array.ofDim[Double](0)
    var outValue: Double = 0d
    var inDepth = false
    var index: Integer = -1

    if (inputValues.length == 1) inDepth = true

    if (init) {
      for (inputValue <- inputValues) {
        index = index + 1
        val inputNode = container.inputNodes(index)
        inputNode.asInstanceOf[InputNeuron].initInput(inputValue, tolerance)
        val outputNode = container.outputNodes(index)
        outputNode.asInstanceOf[OutputNeuron].initOutput(inputValue, tolerance)
      }
    }
    var doContinue: Boolean = false
    index = 0
    for (inputValue <- inputValues) {
      index = index + 1
      doContinue = false
      NeuronCollectionCache.clear()
      for (outputNode <- container.outputNodes) {
        if (!doContinue) {
          outValue = outputNode.collectInConnections(inputValue, false)
          NetTrace.WriteMessage("received outvalue " + index + ": " + outValue + " - searched: " + inputValue)
          if (outValue > inputValue - tolerance && outValue < inputValue + tolerance) {
            outValues = outValues :+ outValue
            NetTrace.WriteMessage("found outvalue " + index + ": " + outValue + " searched: " + inputValue)
            doContinue = true
          }
        }
      }
    }
    outValues
  }
}
