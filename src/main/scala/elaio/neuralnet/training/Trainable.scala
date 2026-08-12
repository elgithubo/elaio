package elaio.neuralnet.training

import java.nio.file.Path
import elaio.neuralnet.attention.AttentionLayer.ForwardPass
import elaio.neuralnet.attention.DepthAttention
import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.bigdata.NeuronNetwork
import elaio.neuralnet.persistence.{NetworkStateMapper, PersistenceAction, PersistenceHandler}
import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.processing.Backpropagation

trait Trainable {

  protected val persistenceAction: Option[PersistenceAction]

  // how many epochs to run
  protected val epochs: Int
  // define the net's learning rate
  protected val learningRate: Double
  // Cap the length of a single update, which is what stops a run from exploding in the first epochs.
  protected val maxUpdateNorm: Double
  // Set to epochs to clip throughout, or to 0 to disable gradient clipping entirely.
  protected val clipUntilEpoch: Int

  // tell the net the wanted answer - only backpropagation reads this, never a forward pass
  protected def initTargets(container: NeuronNetwork, targetValues: Array[Double]): Unit

  private val random = new scala.util.Random
  private val neuronCollectionCache = new NeuronCollectionCache

  // set once the network is built, when attention is wanted
  protected var attention: Option[DepthAttention] = None

  // run the test case
  def run(): Unit

  protected def traceAction(): Unit =
    NetTrace.WriteMessage("no further trace action defined - adjust test class to override this message")

  protected def forwardPass(container: NeuronNetwork): Option[ForwardPass] =
    attention match {
      case Some(depthAttention) =>
        Some(depthAttention.refine(container.reverseOrder, () => container.forward(neuronCollectionCache)))
      case None =>
        container.forward(neuronCollectionCache)
        None
    }

  protected final def processTokens(
      container: NeuronNetwork,
      persistenceAction: Option[PersistenceAction],
      trainingData: => (Array[TokenMatrix], Array[Array[Double]])
  ): Unit = {
    require(attention.isEmpty || persistenceAction.isEmpty, "attention persistence is not supported yet")
    NetTrace.WriteMessage("")
    traceAction()
    NetTrace.WriteMessage("")
    persistenceAction match {
      case Some(PersistenceAction.Load(file)) =>
        load(container, file)

      case _ =>
        // weight initialization has to happen after init(), when every neuron's fan-in is final
        val weightCount = WeightInitializer.initialize(container.reverseOrder)
        NetTrace.WriteMessage("connection weights initialized: " + weightCount)

        val (trainTokens, trainOutputs) = trainingData
        require(trainTokens.length == trainOutputs.length, "each tokenized example needs a result")
        NetTrace.WriteMessage("training on " + trainTokens.length + " examples over " + epochs + " epochs with learning rate " + learningRate)
        NetTrace.WriteMessage("gradient clipping at " + maxUpdateNorm + " for the first " + clipUntilEpoch + " epochs")
        NetTrace.WriteMessage("")
        train(container, trainTokens, trainOutputs)

        persistenceAction match {
          case Some(PersistenceAction.Save(file)) => save(container, file)
          case _                                  => ()
        }
    }
  }

  // summed squared error of the last forward pass against the targets set on the outputs
  private def squaredError(container: NeuronNetwork): Double  = {
    var total = 0d
    for (outputNode <- container.outputNodes) {
      val residual = outputNode.target - outputNode.value
      total = total + residual * residual
    }
    total
  }

  private def load(container: NeuronNetwork, file: Path): Unit = {
    NetTrace.WriteMessage("loading network state from " + file)
    val stateContainer = new PersistenceHandler().load(file)
    NetworkStateMapper.restore(stateContainer, container)
    NetTrace.WriteMessage(
      "loaded " + stateContainer.neuronStore.size + " neurons and " + stateContainer.connectionStore.size + " connections"
    )
  }

  private def save(container: NeuronNetwork, file: Path): Unit = {
    val stateContainer = NetworkStateMapper.capture(container)
    new PersistenceHandler().save(stateContainer, file)
    NetTrace.WriteMessage(
      "saved " + stateContainer.neuronStore.size + " neurons and " + stateContainer.connectionStore.size + " connections to " + file
    )
  }

  // execute the actual training, which is a forward pass followed by backpropagation for each example, repeated for the number of epochs.
  private def train(
      container: NeuronNetwork,
      trainTokens: Array[TokenMatrix],
      trainOutputs: Array[Array[Double]]
  ): Unit = {
    require(trainTokens.length == trainOutputs.length, "need one output for every input")

    val trainingExamples = trainTokens.zip(trainOutputs).toSeq
    for (epoch <- 1 to epochs) {
      // the cap is only in force while the run is still fragile
      val updateNorm = if (epoch <= clipUntilEpoch) maxUpdateNorm else Double.PositiveInfinity
      var totalError = 0d

      // shuffled so the updates do not settle into a fixed cycle
      for ((tokens, targetValues) <- random.shuffle(trainingExamples)) {
        container.initInputs(tokens)
        initTargets(container, targetValues)
        val attentionPass = forwardPass(container)
        totalError = totalError + squaredError(container)
        Backpropagation.run(container.reverseOrder, learningRate, updateNorm)
        for (depthAttention <- attention; pass <- attentionPass)
          depthAttention.applyGradients(pass, learningRate, updateNorm)
      }
      if (epoch == 1 || epoch % 100 == 0 || epoch == epochs)
        NetTrace.WriteMessage("epoch " + epoch + ": total squared error = " + totalError, 1)
      if (epoch == clipUntilEpoch && epoch < epochs)
        NetTrace.WriteMessage("update cap released after epoch " + epoch, 1)
    }
  }
}
