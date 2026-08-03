package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class DivisionTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val tolerance = 0.3d
  override protected val learningRate = 0.015d
  override protected val epochs = 5000
  override protected val clipUntilEpoch = 1000

  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value / 5)
}
