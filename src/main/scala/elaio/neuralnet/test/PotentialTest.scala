package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class PotentialTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val tolerance = 500d
  override protected val epochs = 50000
  override protected val clipUntilEpoch = 50000
  override protected val learningRate = 0.00001d
  override protected val maxUpdateNorm = 50000d

  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * value)
}
