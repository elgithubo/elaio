package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{InputNeuron, NeuronCounter, NeuronDataCreator, OutputNeuron}
import elaio.neuralnet.training.Trainable

trait MathTest extends Trainable {

  protected val epochs = 15000
  protected val maxUpdateNorm = 700d
  protected val clipUntilEpoch = 5000

  protected val tolerance = 1d

  private val dimOuter = 2
  private val width = 5
  private val trainCount = 250
  private val numberOfQuestions = 5

  // the task to learn
  protected def targetOf(inputValues: Array[Double]): Array[Double]

  // define the input values for a single training example
  private def randomInput(random: scala.util.Random): Array[Double] =
    Array.fill(width)(random.nextDouble() * 2000d - 1000d) // random double values in the range [-1000, 1000]

  override def run(): Unit = {
    // enable the following line to write detailed trace messages to stdout, disable it for no output.
    NetTrace.started = true

    NetTrace.WriteMessage("start of test run (if processing diverges with NaN, please rerun)")
    NetTrace.WriteMessage("")
    NetTrace.WriteMessage("build dimension: " + dimOuter)
    NetTrace.WriteMessage("in/out width: " + width)

    val random = new scala.util.Random

    val container = new TensoredContainer(dimOuter, width, new NeuronDataCreator)
    container.init()
    NetTrace.WriteMessage("total neurons created: " + NeuronCounter.counter)

    // process evaluates training data only when training is required
    process(
      container,
      persistenceAction,
      {
        val trainInputs = Array.fill(trainCount)(randomInput(random))
        val trainOutputs = trainInputs.map(targetOf)
        (trainInputs, trainOutputs)
      }
    )

    // the actual test: test inputs the net has never been trained on
    (1 to numberOfQuestions).foreach(_ =>
      NetTrace.WriteMessage("")
      val checkInput = randomInput(random)
      NetTrace.WriteMessage("checking an unseen input: " + checkInput.map(v => f"$v%.3f").mkString(" | "))
      initInputs(container, checkInput)
      // one forward pass with the test values
      forwardPass(container)
      val receivedResult: Array[Double] = container.outputNodes.map(_.value)
      checkOutputs(receivedResult, targetOf(checkInput))
    )

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
}
