package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.{DataCreator, TensoredContainer}
import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.test.Trainable
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.training.Backpropagation
import elaio.neuralnet.units.OutputNeuron

trait Trainable {

  protected val epochs: Int
  
  protected val learningRate: Double
  // Caps the length of a single update, which is what stops a run from exploding in
  // the first epochs.
  protected val maxUpdateNorm: Double
  // Set to epochs to clip throughout, or to 0 to disable clipping entirely.
  protected val clipUntilEpoch: Int

  protected def initInputs(container: elaio.neuralnet.bigdata.container.TensoredContainer, inputValues: Array[Double]): Unit
  protected def initTargets(container: TensoredContainer, targetValues: Array[Double]): Unit
  protected def targetOf(inputValues: Array[Double]): Array[Double]
  protected def run(): Unit

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
      random: scala.util.Random
  ): Unit = {
    for (epoch <- 1 to epochs) {
      // the cap is only in force while the run is still fragile
      val updateNorm = if (epoch <= clipUntilEpoch) maxUpdateNorm else Double.PositiveInfinity
      var totalError = 0d
      // shuffled so the updates do not settle into a fixed cycle
      for (inputValues <- random.shuffle(trainInputs.toSeq)) {
        initInputs(container, inputValues)
        initTargets(container, targetOf(inputValues))
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
