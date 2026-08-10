package elaio.neuralnet.test

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.persistence.PersistenceAction
import elaio.neuralnet.trace.NetTrace

final class SubtractionTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val learningRate = 0.005d
  override protected val maxUpdateNorm = 1000d

  protected def targetOf(tokens: TokenMatrix): Array[Double] = tokens.flatten.map(value => value - 13)

  override protected def traceAction(): Unit =
    NetTrace.WriteMessage("testing: x - 13")
}
