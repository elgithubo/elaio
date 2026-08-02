package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class DivisionTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override val tolerance = 0.3d
  override protected val learningRate = 0.015
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value / 5)
}
