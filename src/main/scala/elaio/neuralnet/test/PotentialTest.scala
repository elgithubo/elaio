package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class PotentialTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override val tolerance = 50d
  override val epochs = 50000
  override val clipUntilEpoch = 50000
  override protected val learningRate = 0.00015
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * value)
}
