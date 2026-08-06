package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class PotentialTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val tolerance = 2000d
  override protected val epochs = 30000
  override protected val clipUntilEpoch = 30000
  override protected val learningRate = 0.00001d
  override protected val maxUpdateNorm = 50000d

  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * value)
}
