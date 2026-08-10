package elaio.neuralnet.test

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.persistence.PersistenceAction
import elaio.neuralnet.trace.NetTrace

final class MultiplicationTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val tolerance = 2d
  override protected val learningRate = 0.002d
  override protected val clipUntilEpoch = 10000
  override protected val maxUpdateNorm = 2000d

  protected def targetOf(tokens: TokenMatrix): Array[Double] = tokens.flatten.map(value => value * 3)

  override protected def traceAction(): Unit =
    NetTrace.WriteMessage("testing: x * 3")
}
