package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class SubtractionTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val learningRate = 0.003d

  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value - 13)
}
