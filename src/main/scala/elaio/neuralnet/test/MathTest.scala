package elaio.neuralnet.test

import elaio.neuralnet.bigdata.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{HiddenNeuronLeakyRelu, HiddenNeuronSquare, InputNeuron, NeuronDataCreator, OutputNeuron}
import elaio.neuralnet.training.Trainable

trait MathTest extends Trainable {

  override protected val epochs = 15000
  override protected val clipUntilEpoch = 7500

  protected val tolerance = 1d

  private val dimOuter = 2
  private val width = 5
  private val trainCount = 250
  protected val numberOfQuestions = 5

  // the task to learn
  protected def targetOf(inputValues: Array[Double]): Array[Double]

  // how an input reads in the log - overridden where the channels are not all data
  protected def describeInput(inputValues: Array[Double]): String =
    inputValues.map(v => f"$v%.3f").mkString(" | ")

  // define the input values for a single training example
  protected def randomInput(random: scala.util.Random): Array[Double] =
    Array.fill(width)(random.nextDouble() * 2000d - 1000d) // random double values in the range [-1000, 1000]

  // the questions asked after training - overridden where they should be grouped
  protected def checkInputs(random: scala.util.Random): Seq[Array[Double]] =
    Seq.fill(numberOfQuestions)(randomInput(random))

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
    val neurons = container.reverseOrder.sequence
    NetTrace.WriteMessage("total neurons created: " + neurons.length)
    NetTrace.WriteMessage("input neurons: " + neurons.count(_.isInstanceOf[InputNeuron]), 1)
    NetTrace.WriteMessage("hidden square neurons: " + neurons.count(_.isInstanceOf[HiddenNeuronSquare]), 1)
    NetTrace.WriteMessage("hidden leaky relu neurons: " + neurons.count(_.isInstanceOf[HiddenNeuronLeakyRelu]), 1)
    NetTrace.WriteMessage("output neurons: " + neurons.count(_.isInstanceOf[OutputNeuron]), 1)

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
    checkInputs(random).foreach(checkInput =>
      NetTrace.WriteMessage("")
      NetTrace.WriteMessage("checking an unseen input: " + describeInput(checkInput))
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
    // one line per output, fixed width, so a block can be scanned at a glance
    for (index <- outValues.indices) {
      val off = math.abs(expected(index) - outValues(index))
      val hit = off < tolerance
      if (hit) within = within + 1
      NetTrace.WriteMessage(
        f"output ${index + 1}: ${outValues(index)}%13.3f   target ${expected(index)}%13.3f" +
          f"   off ${off}%9.3f   ${if (hit) "hit" else "-"}",
        1
      )
    }
    NetTrace.WriteMessage(
      "unseen input: " + within + " of " + outValues.length + " outputs within tolerance (" + tolerance + ")"
    )
  }
}
