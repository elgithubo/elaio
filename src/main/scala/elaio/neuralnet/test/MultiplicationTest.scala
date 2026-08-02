package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

final class MultiplicationTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override val tolerance = 2d
  override protected val learningRate = 0.0015
  override protected val clipUntilEpoch = 10000
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * 3)
}
