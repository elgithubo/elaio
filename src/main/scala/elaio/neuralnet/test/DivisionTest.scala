package elaio.neuralnet.test

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.persistence.PersistenceAction
import elaio.neuralnet.trace.NetTrace

final class DivisionTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val tolerance = 0.3d
  override protected val learningRate = 0.015d
  override protected val epochs = 5000
  override protected val clipUntilEpoch = epochs
  override protected val maxUpdateNorm = 700d

  override protected val layeredTokens = true
  override protected val tokenWidth = 5

  protected def targetOf(tokens: TokenMatrix): Array[Double] = tokens.flatten.map(value => value / 5)

  override protected def traceAction(): Unit =
    NetTrace.WriteMessage("testing: x / 5")
}
