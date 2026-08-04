package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class MultiplicationTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val tolerance = 2d
  override protected val learningRate = 0.002d
  override protected val clipUntilEpoch = 10000
  override protected val maxUpdateNorm = 2000d

  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * 3)
}
