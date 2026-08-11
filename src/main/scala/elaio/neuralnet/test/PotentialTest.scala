package elaio.neuralnet.test

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.persistence.PersistenceAction
import elaio.neuralnet.trace.NetTrace

final class PotentialTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  //private val keyCount = 3
  //private val keyTokenWidth = 1 // keyCount + 2 // query marker, one-hot key, value
  //private val keyTokenCount = 1 //keyCount + 1

  override protected val tolerance = 2000d
  override protected val learningRate = 0.00001d
  override protected val maxUpdateNorm = 50000d
  override protected val attentionEnabled: Boolean = true

  override protected def tokenWidth: Int = 5
  override protected val dimOuter = 2
  override protected val inWidth = 5
  override protected val inputMinimum = -1d
  override protected val inputMaximum = 1d
  override protected val trainCount = 300
  override protected val numberOfQuestions = 300
  override protected val epochs = 25000
  override protected val clipUntilEpoch = epochs
  //override protected val learningRate = 0.005d
  //override protected val maxUpdateNorm = 100d

  override protected val tokenFactor = 1000d

  protected def targetOf(tokens: TokenMatrix): Array[Double] = tokens.flatten.map(value => value * value)

  override protected def traceAction(): Unit =
    NetTrace.WriteMessage("testing: x * x")
}
