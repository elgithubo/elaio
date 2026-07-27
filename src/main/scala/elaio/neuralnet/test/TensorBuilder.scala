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

  // the task, stated only here - everything downstream reads it back off the OutputNeuron
  private def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(_ * 2d)

  def run(): Unit = {
    // Enable the following line to write detailed trace messages to stdout
    NetTrace.started_(true)

    NetTrace.WriteMessage("start of test run")

    NetTrace.WriteMessage("build dimension: " + dimOuter )
    NetTrace.WriteMessage{"in/out width " + width }

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

    // How well the training examples come back - a poor number means training did not finish
    var trainedWithin = 0
    for (inputValues <- trainInputs) {
      initInputs(container, inputValues)
      forwardPass(container)
      trainedWithin = trainedWithin + countWithinTolerance(container, targetOf(inputValues))
    }
    NetTrace.WriteMessage(
      "trained inputs: " + trainedWithin + " of " + (trainInputs.length * width) + " outputs within tolerance"
    )

    // the actual test: an input the net has never been trained on
    NetTrace.WriteMessage("")
    NetTrace.WriteMessage("checking an unseen input: " + checkInput.map(v => f"$v%.3f").mkString(", "))
    initInputs(container, checkInput)
    val checkOutValues: Array[Double] = feedbackIn(container)
    val withinTolerance: Array[Double] = checkOutputs(checkOutValues, targetOf(checkInput))
    NetTrace.WriteMessage(
      "unseen input: " + withinTolerance.length + " of " + width + " outputs within tolerance"
    )

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
    for (outputNode <- container.outputNodes)
      outputNode.collectInConnections()
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

  // scores the last forward pass against expected values passed in, not against the outputs' targets
  private def countWithinTolerance(container: TensoredContainer, expected: Array[Double]): Int =
    container.outputNodes.indices.count(index =>
      math.abs(expected(index) - container.outputNodes(index).value) < tolerance
    )

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

  // One forward pass, returning what the net answered. Takes no expected values.
  private def feedbackIn(container: TensoredContainer): Array[Double] = {
    forwardPass(container)
    NetTrace.WriteMessage("distinct neurons visited this pass: " + NeuronCollectionCache.size)
    container.outputNodes.map(_.value)
  }

  // Scores answers the net already gave, returning those within tolerance
  private def checkOutputs(outValues: Array[Double], expected: Array[Double]): Array[Double] = {
    require(outValues.length == expected.length, "need one expected value per output")
    var within: Array[Double] = Array.ofDim[Double](0)
    for (index <- outValues.indices) {
      NetTrace.WriteMessage("received outvalue " + (index + 1) + ": " + outValues(index) + " - searched: " + expected(index))
      if (math.abs(expected(index) - outValues(index)) < tolerance) {
        within = within :+ outValues(index)
        NetTrace.WriteMessage("found outvalue " + (index + 1) + ": " + outValues(index) + " searched: " + expected(index))
      }
    }
    within
  }
}
