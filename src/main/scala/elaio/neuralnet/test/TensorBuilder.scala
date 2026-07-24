package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{InputNeuron, NeuronCounter, NeuronDataCreator, OutputNeuron}
import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.training.Backpropagation

object TensorBuilder {
  def run(): Unit = {
    // Enable the following line to write detailed trace messages to stdout
    NetTrace.started_(true)

    NetTrace.WriteMessage("start of test run")

    val neuronDataCreatorTensored = new NeuronDataCreator

    val container = new TensoredContainer(
      5,
      4,
      neuronDataCreatorTensored,
      true,
    )
    container.init()
    NetTrace.WriteMessage("total neurons created: " + NeuronCounter.current)

    val inputValues = Array(6d, 5d, 4d, 3d)
    val tolerance = 0.1d
    initInputsOutputs(container, inputValues, tolerance)
    train(container, learningRate = 0.1d, epochs = 200)

    val outValues: Array[Double] = feedbackIn(container, inputValues, tolerance)
    for (outValue <- outValues) {
      NetTrace.WriteMessage("outValue: " + outValue)
    }

    NetTrace.WriteMessage("end of test run")
  }

  private def initInputsOutputs(container: TensoredContainer, inputValues: Array[Double], tolerance: Double): Unit = {
    for (index <- inputValues.indices) {
      container.inputNodes(index).asInstanceOf[InputNeuron].initInput(inputValues(index), tolerance)
      container.outputNodes(index).asInstanceOf[OutputNeuron].initOutput(inputValues(index), tolerance)
    }
  }

  private def train(container: TensoredContainer, learningRate: Double, epochs: Int): Unit = {
    for (epoch <- 1 to epochs) {
      NeuronCollectionCache.clear()
      var totalError = 0d
      for (outputNode <- container.outputNodes) {
        outputNode.collectInConnections(0d, false)
        val residual = outputNode.asInstanceOf[OutputNeuron].target - outputNode.value
        totalError = totalError + residual * residual
      }
      Backpropagation.run(container.outputNodes, learningRate)
      if (epoch == 1 || epoch % 20 == 0 || epoch == epochs)
        NetTrace.WriteMessage("epoch " + epoch + ": total squared error = " + totalError)
    }
  }

  private def feedbackIn(container: TensoredContainer, inputValues: Array[Double], tolerance: Double): Array[Double] = {
    var outValues: Array[Double] = Array.ofDim[Double](0)
    var outValue: Double = 0d
    var index: Integer = 0

    var doContinue: Boolean = false
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
      NetTrace.WriteMessage("distinct neurons visited this pass: " + NeuronCollectionCache.size)
    }
    outValues
  }
}
