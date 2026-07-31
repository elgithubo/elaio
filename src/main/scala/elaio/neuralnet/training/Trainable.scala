package elaio.neuralnet.training

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.OutputNeuron

trait Trainable {

  // how many epochs to run
  protected val epochs: Int
  // define the net's learning rate
  protected val learningRate: Double
  // Cap the length of a single update, which is what stops a run from exploding in the first epochs.
  protected val maxUpdateNorm: Double
  // Set to epochs to clip throughout, or to 0 to disable clipping entirely.
  protected val clipUntilEpoch: Int

  // ask the net a question
  protected def initInputs(container: TensoredContainer, inputValues: Array[Double]): Unit
  // tell the net the wanted answer - only backpropagation reads this, never a forward pass
  protected def initTargets(container: TensoredContainer, targetValues: Array[Double]): Unit

  // run the test case
  def run(): Unit

  protected def forwardPass(container: TensoredContainer): Unit = {
    NeuronCollectionCache.clear()
    for (outputNode <- container.outputNodes) outputNode.collectInConnections()
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

  // execute the actual training, which is a forward pass followed by backpropagation for each example, repeated for the number of epochs.
  protected def train(
      container: TensoredContainer,
      trainInputs: Array[Array[Double]],
      trainOutputs: Array[Array[Double]],
      random: scala.util.Random,
  ): Unit = {
    require(
      trainInputs.length == trainOutputs.length,
      "need one output for every input"
    )

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
      if (epoch == clipUntilEpoch && clipUntilEpoch < epochs)
        NetTrace.WriteMessage("update cap released after epoch " + epoch, 1)
    }
  }
}
