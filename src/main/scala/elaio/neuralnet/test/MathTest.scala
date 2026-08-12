package elaio.neuralnet.test

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.attention.DepthAttention
import elaio.neuralnet.bigdata.{LayeredContainer, NeuronNetwork}
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.{HiddenNeuronLeakyRelu, HiddenNeuronSquare, InputNeuron, NeuronDataCreator, IntermediateOutputNeuron}
import elaio.neuralnet.training.Trainable

trait MathTest extends Trainable {

  override protected val epochs = 15000
  override protected val clipUntilEpoch = 7500

  protected val tolerance = 1d

  protected val dimOuter = 2
  protected val inWidth = 5
  protected val inputMinimum = -1000d
  protected val inputMaximum = 1000d
  protected val trainCount = 250
  protected val numberOfQuestions = 5
  protected val attentionEnabled = false
  protected val externalOutWidth = 5

  // how wide one token's representation is where the read-out picks it up
  protected def tokenOutWidth: Int = tokenWidth

  // An example is tokenCount rows of tokenWidth values. By default a single token carries the
  // whole input; a task with real token structure overrides tokenWidth. Both are defs, because
  // a val here would read inWidth before an overriding subclass has assigned it.
  protected def tokenWidth: Int = inWidth
  protected final def tokenCount: Int = {
    require(tokenWidth > 0 && inWidth % tokenWidth == 0, "input width must be divisible by a positive token width")
    inWidth / tokenWidth
  }

  protected val tokenFactor = 1d

  // the task to learn
  protected def targetOf(tokens: TokenMatrix): Array[Double]

  // how an input reads in the log - overridden where the channels are not all data
  protected def describeInput(tokens: TokenMatrix): String =
    tokens.map(_.map(v => f"$v%.3f").mkString(" | ")).mkString("  ||  ")

  protected def randomValue(random: scala.util.Random): Double =
    random.nextDouble() * (inputMaximum - inputMinimum) + inputMinimum

  // define the tokens of a single training example
  protected def randomTokens(random: scala.util.Random): TokenMatrix =
    Array.fill(tokenCount)(Array.fill(tokenWidth)(randomValue(random) * tokenFactor))

  protected def trainingTokens(random: scala.util.Random): Array[TokenMatrix] =
    Array.fill(trainCount)(randomTokens(random))

  // the questions asked after training - overridden where they should be grouped
  protected def checkTokens(random: scala.util.Random): Seq[TokenMatrix] =
    Seq.fill(numberOfQuestions)(randomTokens(random))

  override def run(): Unit = {
    // enable the following line to write detailed trace messages to stdout, disable it for no output.
    NetTrace.started = true

    NetTrace.WriteMessage("start of test run (if processing diverges with NaN, please rerun)")
    NetTrace.WriteMessage("")
    NetTrace.WriteMessage("build dimension: " + dimOuter)
    NetTrace.WriteMessage("input width: " + inWidth + " (" + tokenCount + " tokens of " + tokenWidth + ")")
    NetTrace.WriteMessage("output width: " + externalOutWidth)
    NetTrace.WriteMessage( s"containers: $tokenCount stacked, one per token" )
    NetTrace.WriteMessage("attention across depths: " + attentionEnabled)

    val random = new scala.util.Random

    val container = buildNetwork()
    val neurons = container.reverseOrder.sequence
    NetTrace.WriteMessage("total neurons created: " + neurons.length)
    NetTrace.WriteMessage("input neurons: " + neurons.count(_.isInstanceOf[InputNeuron]), 1)
    NetTrace.WriteMessage("hidden square neurons: " + neurons.count(_.isInstanceOf[HiddenNeuronSquare]), 1)
    NetTrace.WriteMessage("hidden leaky relu neurons: " + neurons.count(_.isInstanceOf[HiddenNeuronLeakyRelu]), 1)
    NetTrace.WriteMessage("intermediate output neurons: " + neurons.count(_.isInstanceOf[IntermediateOutputNeuron]), 1)
    NetTrace.WriteMessage("output neurons: " + container.outputNodes.length, 1)

    // processTokens evaluates training data only when training is required
    processTokens(
      container,
      persistenceAction,
      {
        val trainTokens = trainingTokens(random)
        (trainTokens, trainTokens.map(targetOf))
      }
    )

    // the actual test: test inputs the net has never been trained on
    checkTokens(random).foreach(checkToken =>
      NetTrace.WriteMessage("")
      NetTrace.WriteMessage("checking an unseen input: " + describeInput(checkToken))
      container.initInputs(checkToken)
      // one forward pass with the test values
      forwardPass(container)
      val receivedResult: Array[Double] = container.outputNodes.map(_.value)
      checkOutputs(receivedResult, targetOf(checkToken))
    )

    NetTrace.WriteMessage("")
    NetTrace.WriteMessage("end of test run")
  }

  // One container per token, or a stack of one carrying the whole input. The stack of one is built
  // without a read-out rank, so it is the same graph a bare TensoredContainer builds - which is why
  // both cases can go through the same class.
  private def buildNetwork(): NeuronNetwork = {
    val layered =
      new LayeredContainer(dimOuter, tokenCount, tokenWidth, tokenOutWidth, externalOutWidth, new NeuronDataCreator)
    layered.init()
    // attention is bound to the exact graph it was built for
    if (attentionEnabled) attention = Some(new DepthAttention(layered.reverseOrder))
    layered
  }

  protected def initTargets(container: NeuronNetwork, targetValues: Array[Double]): Unit = {
    require(
      targetValues.length == externalOutWidth,
      "expected " + externalOutWidth + " targets but got " + targetValues.length
    )
    for (index <- targetValues.indices)
      container.outputNodes(index).initOutput(targetValues(index))
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
