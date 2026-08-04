package elaio.neuralnet.training

import java.nio.file.Path
import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.persistence.{NetworkStateMapper, PersistenceAction, PersistenceHandler}
import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.OutputNeuron
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

  // ask the net a question
  protected def initInputs(container: TensoredContainer, inputValues: Array[Double]): Unit
  // tell the net the wanted answer - only backpropagation reads this, never a forward pass
  protected def initTargets(container: TensoredContainer, targetValues: Array[Double]): Unit

  private val random = new scala.util.Random

  // run the test case
  def run(): Unit

  protected def forwardPass(container: TensoredContainer): Unit = {
    NeuronCollectionCache.clear()
    for (outputNode <- container.outputNodes) outputNode.collectInConnections()
  }

  protected final def process(
      container: TensoredContainer,
      persistenceAction: Option[PersistenceAction],
      trainingData: => (Array[Array[Double]], Array[Array[Double]])
  ): Unit = persistenceAction match {
    case Some(PersistenceAction.Load(file)) =>
      load(container, file)

    case _ =>
      // weight initialization has to happen after init(), when every neuron's fan-in is final
      val weightCount = WeightInitializer.initialize(container.outputNodes)
      NetTrace.WriteMessage("connection weights initialized: " + weightCount)

      val (trainInputs, trainOutputs) = trainingData
      NetTrace.WriteMessage("training on " + trainInputs.length + " examples over " + epochs + " epochs with learning rate " + learningRate)
      NetTrace.WriteMessage("gradient clipping at " + maxUpdateNorm + " for the first " + clipUntilEpoch + " epochs")
      NetTrace.WriteMessage("")
      train(container, trainInputs, trainOutputs)

      persistenceAction match {
        case Some(PersistenceAction.Save(file)) => save(container, file)
        case _                                  => ()
      }
  }

  // summed squared error of the last forward pass against the targets set on the outputs
  private def squaredError(container: TensoredContainer): Double  = {
    var total = 0d
    for (outputNode <- container.outputNodes) {
      val residual = outputNode.asInstanceOf[OutputNeuron].target - outputNode.value
      total = total + residual * residual
    }
    total
  }

  private def load(container: TensoredContainer, file: Path): Unit = {
    NetTrace.WriteMessage("loading network state from " + file)
    val stateContainer = new PersistenceHandler().load(file)
    NetworkStateMapper.restore(stateContainer, container)
    NetTrace.WriteMessage(
      "loaded " + stateContainer.neuronStore.size + " neurons and " + stateContainer.connectionStore.size + " connections"
    )
  }

  private def save(container: TensoredContainer, file: Path): Unit = {
    val stateContainer = NetworkStateMapper.capture(container)
    new PersistenceHandler().save(stateContainer, file)
    NetTrace.WriteMessage(
      "saved " + stateContainer.neuronStore.size + " neurons and " + stateContainer.connectionStore.size + " connections to " + file
    )
  }

  // execute the actual training, which is a forward pass followed by backpropagation for each example, repeated for the number of epochs.
  private def train(
      container: TensoredContainer,
      trainInputs: Array[Array[Double]],
      trainOutputs: Array[Array[Double]]
  ): Unit = {
    require(trainInputs.length == trainOutputs.length, "need one output for every input")

    val trainingExamples = trainInputs.zip(trainOutputs).toSeq
    for (epoch <- 1 to epochs) {
      // the cap is only in force while the run is still fragile
      val updateNorm = if (epoch <= clipUntilEpoch) maxUpdateNorm else Double.PositiveInfinity
      var totalError = 0d

      // shuffled so the updates do not settle into a fixed cycle
      for ((inputValues, targetValues) <- random.shuffle(trainingExamples)) {
        initInputs(container, inputValues)
        initTargets(container, targetValues)
        forwardPass(container)
        totalError = totalError + squaredError(container)
        Backpropagation.run(container.outputNodes, learningRate, updateNorm)
      }
      if (epoch == 1 || epoch % 100 == 0 || epoch == epochs)
        NetTrace.WriteMessage("epoch " + epoch + ": total squared error = " + totalError, 1)
      if (epoch == clipUntilEpoch && epoch < epochs)
        NetTrace.WriteMessage("update cap released after epoch " + epoch, 1)
    }
  }
}
