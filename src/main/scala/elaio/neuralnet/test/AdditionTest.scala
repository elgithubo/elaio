package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class AdditionTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val learningRate = 0.005d
  override protected val maxUpdateNorm = 1000d

  protected def targetOf(tokens: TokenMatrix): Array[Double] = tokens.flatten.map(value => value + 23)
}
