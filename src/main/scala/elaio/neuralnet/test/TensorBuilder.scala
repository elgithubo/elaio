package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{InputNeuron, NeuronCounter, NeuronDataCreator, OutputNeuron}
import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.training.{Backpropagation, WeightInitializer}

object TensorBuilder {

  private val dimOuter = 3
  private val width = 6
  private val trainCount = 200
  private val learningRate = 0.001d
  private val epochs = 50000
  private val tolerance = 0.2d

  // the task, stated only here
  private def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(_ * 2d)

  // run the test case
  def run(): Unit = {
    // Enable the following line to write detailed trace messages to stdout
    NetTrace.started_(true)

    NetTrace.WriteMessage("start of test run")

    NetTrace.WriteMessage("build dimension: " + dimOuter)
    NetTrace.WriteMessage("in/out width: " + width)

    val container = new TensoredContainer(dimOuter, width, new NeuronDataCreator)
    container.init()
    NetTrace.WriteMessage("total neurons created: " + NeuronCounter.current)

    // has to happen after init(), when every neuron's fan-in is final
    val weightCount = WeightInitializer.initialize(container.outputNodes)
    NetTrace.WriteMessage("connection weights initialized: " + weightCount)

    // Needs many examples: a 6 -> 6 map has 36 unknowns and each example gives 6
    // equations. Accuracy on unseen inputs comes from the example count, not the
    // epoch count - 30 examples missed the tolerance entirely, 100 hit 5 of 6.
    val random = new scala.util.Random
    val trainInputs = Array.fill(trainCount)(randomInput(random))
    val checkInput = randomInput(random)

    NetTrace.WriteMessage("training on " + trainInputs.length + " examples over " + epochs + " epochs with learning rate " + learningRate)
    train(container, trainInputs, random)

    // the actual test: an input the net has never been trained on
    NetTrace.WriteMessage("")
    NetTrace.WriteMessage("checking an unseen input: " + checkInput.map(v => f"$v%.3f").mkString(", "))
    initInputs(container, checkInput)
    // One forward pass with the test values
    forwardPass(container)
    NetTrace.WriteMessage("distinct neurons visited this pass: " + NeuronCollectionCache.size)
    val checkOutValues: Array[Double] = container.outputNodes.map(_.value)
    checkOutputs(checkOutValues, targetOf(checkInput))

    NetTrace.WriteMessage("end of test run")
  }

  private def randomInput(random: scala.util.Random): Array[Double] =
    Array.fill(width)(random.nextDouble() * 12d - 6d)

  // asks the net a question
  private def initInputs(container: TensoredContainer, inputValues: Array[Double]): Unit = {
    require(inputValues.length == width, "expected " + width + " inputs but got " + inputValues.length)
    for (index <- inputValues.indices)
      container.inputNodes(index).asInstanceOf[InputNeuron].initInput(inputValues(index))
  }

  // tells the net the wanted answer - only backpropagation reads this, never a forward pass
  private def initTargets(container: TensoredContainer, targetValues: Array[Double]): Unit = {
    require(targetValues.length == width, "expected " + width + " targets but got " + targetValues.length)
    for (index <- targetValues.indices)
      container.outputNodes(index).asInstanceOf[OutputNeuron].initOutput(targetValues(index))
  }

  // one forward pass over the graph
  private def forwardPass(container: TensoredContainer): Unit = {
    NeuronCollectionCache.clear()
    for (outputNode <- container.outputNodes) outputNode.collectInConnections()
  }

  // summed squared error of the last forward pass against the targets set on the outputs
  private def squaredError(container: TensoredContainer): Double = {
    var total = 0d
    for (outputNode <- container.outputNodes) {
      val residual = outputNode.asInstanceOf[OutputNeuron].target - outputNode.value
      total = total + residual * residual
    }
    total
  }

  // Execute the actual training, which is a forward pass followed by backpropagation for each example, repeated for the number of epochs.
  private def train(
      container: TensoredContainer,
      trainInputs: Array[Array[Double]],
      random: scala.util.Random
  ): Unit = {
    for (epoch <- 1 to epochs) {
      var totalError = 0d
      // shuffled so the updates do not settle into a fixed cycle
      for (inputValues <- random.shuffle(trainInputs.toSeq)) {
        initInputs(container, inputValues)
        initTargets(container, targetOf(inputValues))
        forwardPass(container)
        totalError = totalError + squaredError(container)
        Backpropagation.run(container.outputNodes, learningRate)
      }
      if (epoch == 1 || epoch % 1000 == 0 || epoch == epochs)
        NetTrace.WriteMessage("epoch " + epoch + ": total squared error = " + totalError)
    }
  }

  // Scores answers the net already gave and reports how many are within tolerance
  private def checkOutputs(outValues: Array[Double], expected: Array[Double]): Unit = {
    require(outValues.length == expected.length, "need one expected value per output")
    var within: Int = 0
    for (index <- outValues.indices) {
      NetTrace.WriteMessage("received outvalue " + (index + 1) + ": " + outValues(index) + " - searched: " + expected(index))
      if (math.abs(expected(index) - outValues(index)) < tolerance) {
        within = within + 1
        NetTrace.WriteMessage("found outvalue " + (index + 1) + ": " + outValues(index) + " searched: " + expected(index))
      }
    }
    NetTrace.WriteMessage(
      "unseen input: " + within + " of " + outValues.length + " outputs within tolerance"
    )
  }
}
