package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{InputNeuron, NeuronCounter, NeuronDataCreator, OutputNeuron}
import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.training.{Backpropagation, WeightInitializer}

object TensorBuilder {
  def run(): Unit = {
    // Enable the following line to write detailed trace messages to stdout
    NetTrace.started_(true)

    NetTrace.WriteMessage("start of test run")

    val neuronDataCreatorTensored = new NeuronDataCreator

    val container = new TensoredContainer(
      5,
      6,
      neuronDataCreatorTensored,
    )
    container.init()
    NetTrace.WriteMessage("total neurons created: " + NeuronCounter.current)

    // has to happen after init(), when every neuron's fan-in is final
    val weightCount = WeightInitializer.initialize(container.outputNodes)
    NetTrace.WriteMessage("connection weights initialized: " + weightCount)

    // the task: what the net is asked to turn the inputs into. Stated here and
    // nowhere else - everything downstream reads the target back off the
    // OutputNeuron, so changing the task means changing only this line.
    val inputValues = Array(6d, 5d, 4d, 3d, 0.5d, -6d)
    val targetValues = inputValues.map(_ * 2)
    val tolerance = 0.1d

    initInputsOutputs(container, inputValues, targetValues)
    train(container, learningRate = 0.003, epochs = 5000)

    val outValues: Array[Double] = feedbackIn(container, tolerance)
    for (outValue <- outValues) {
      NetTrace.WriteMessage("outValue: " + outValue)
    }

    NetTrace.WriteMessage("end of test run")
  }

  private def initInputsOutputs(
      container: TensoredContainer,
      inputValues: Array[Double],
      targetValues: Array[Double]
  ): Unit = {
    require(
      inputValues.length == targetValues.length,
      "every input needs its own target: " + inputValues.length + " inputs but " + targetValues.length + " targets"
    )
    for (index <- inputValues.indices) {
      container.inputNodes(index).asInstanceOf[InputNeuron].initInput(inputValues(index))
      container.outputNodes(index).asInstanceOf[OutputNeuron].initOutput(targetValues(index))
    }
  }

  private def train(container: TensoredContainer, learningRate: Double, epochs: Int): Unit = {
    for (epoch <- 1 to epochs) {
      NeuronCollectionCache.clear()
      var totalError = 0d
      for (outputNode <- container.outputNodes) {
        outputNode.collectInConnections()
        val residual = outputNode.asInstanceOf[OutputNeuron].target - outputNode.value
        totalError = totalError + residual * residual
      }
      Backpropagation.run(container.outputNodes, learningRate)
      if (epoch == 1 || epoch % 20 == 0 || epoch == epochs)
        NetTrace.WriteMessage("epoch " + epoch + ": total squared error = " + totalError)
    }
  }

  // Runs a forwards pass and validates every output against the target it was initialised with.
  // Returns the values that landed within tolerance.
  private def feedbackIn(container: TensoredContainer, tolerance: Double): Array[Double] = {
    var outValues: Array[Double] = Array.ofDim[Double](0)
    var index: Integer = 0

    NeuronCollectionCache.clear()
    for (outputNode <- container.outputNodes) {
      index = index + 1
      val outValue = outputNode.collectInConnections()
      val target = outputNode.asInstanceOf[OutputNeuron].target
      NetTrace.WriteMessage("received outvalue " + index + ": " + outValue + " - searched: " + target)
      if (math.abs(target - outValue) < tolerance) {
        outValues = outValues :+ outValue
        NetTrace.WriteMessage("found outvalue " + index + ": " + outValue + " searched: " + target)
      }
    }
    NetTrace.WriteMessage("distinct neurons visited this pass: " + NeuronCollectionCache.size)
    outValues
  }
}
