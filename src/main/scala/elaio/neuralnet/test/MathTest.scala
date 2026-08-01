package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{InputNeuron, NeuronCounter, NeuronDataCreator, OutputNeuron}
import elaio.neuralnet.training.{Trainable, WeightInitializer}

trait MathTest extends Trainable {

  protected val learningRate: Double
  protected val epochs = 15000
  protected val maxUpdateNorm = 700d
  protected val clipUntilEpoch = 1000

  protected val tolerance = 1d

  private val dimOuter = 3
  private val width = 5
  private val trainCount = 250

  // the task to learn
  protected def targetOf(inputValues: Array[Double]): Array[Double]

  def run(): Unit = {
    // enable the following line to write detailed trace messages to stdout, disable it for no output.
    NetTrace.started = true

    NetTrace.WriteMessage("start of test run (if processing diverges with NaN, please rerun)")
    NetTrace.WriteMessage("")
    NetTrace.WriteMessage("build dimension: " + dimOuter)
    NetTrace.WriteMessage("in/out width: " + width)

    val container = new TensoredContainer(dimOuter, width, new NeuronDataCreator)
    container.init()
    NetTrace.WriteMessage("total neurons created: " + NeuronCounter.current.toInt)

    // weight initialization has to happen after init(), when every neuron's fan-in is final
    val weightCount = WeightInitializer.initialize(container.outputNodes)
    NetTrace.WriteMessage("connection weights initialized: " + weightCount.toInt)

    val random = new scala.util.Random
    // generate training data, which is just random inputs and the corresponding calculated outputs
    val trainInputs = Array.fill(trainCount)(randomInput(random))
    val trainOutputs = trainInputs.map(targetOf)

    //execute the training, which is a forward pass followed by backpropagation for each example, repeated for the number of epochs.
    NetTrace.WriteMessage("training on " + trainInputs.length + " examples over " + epochs + " epochs with learning rate " + learningRate)
    NetTrace.WriteMessage("update norm capped at " + maxUpdateNorm + " for the first " + clipUntilEpoch + " epochs")
    NetTrace.WriteMessage("")
    train(container, trainInputs, trainOutputs)

    // the actual test: an input the net has never been trained on
    NetTrace.WriteMessage("")
    val checkInput = randomInput(random)
    NetTrace.WriteMessage("checking an unseen input: " + checkInput.map(v => f"$v%.3f").mkString(", "))
    initInputs(container, checkInput)
    // one forward pass with the test values
    forwardPass(container)
    NetTrace.WriteMessage("")
    val receivedResult: Array[Double] = container.outputNodes.map(_.value)
    checkOutputs(receivedResult, targetOf(checkInput))

    NetTrace.WriteMessage("")
    NetTrace.WriteMessage("end of test run")
  }

  protected def initInputs(container: TensoredContainer, inputValues: Array[Double]): Unit = {
    require(inputValues.length == width, "expected " + width + " inputs but got " + inputValues.length)
    for (index <- inputValues.indices)
      container.inputNodes(index).asInstanceOf[InputNeuron].initInput(inputValues(index))
  }

  protected def initTargets(container: TensoredContainer, targetValues: Array[Double]): Unit = {
    require(targetValues.length == width, "expected " + width + " targets but got " + targetValues.length)
    for (index <- targetValues.indices)
      container.outputNodes(index).asInstanceOf[OutputNeuron].initOutput(targetValues(index))
  }

  private def checkOutputs(outValues: Array[Double], expected: Array[Double]): Unit = {
    require(outValues.length == expected.length, "need one expected value per output")
    var within: Int = 0
    for (index <- outValues.indices) {
      NetTrace.WriteMessage("received outvalue " + (index + 1) + ": " + outValues(index) + " - searched: " + expected(index))
      if (math.abs(expected(index) - outValues(index)) < tolerance) {
        within = within + 1
        NetTrace.WriteMessage("found outvalue", 1)
      }
    }
    NetTrace.WriteMessage(
      "unseen input: " + within + " of " + outValues.length + " outputs within tolerance (" + tolerance + ")"
    )
  }

  private def randomInput(random: scala.util.Random): Array[Double] =
    Array.fill(width)(random.nextDouble() * 200d - 100d) // random double values in the range [-100, 100]

}
