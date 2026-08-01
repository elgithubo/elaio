package elaio.neuralnet.test

object PotentialTest extends MathTest {
  override val tolerance = 50d
  override val epochs = 50000
  override val clipUntilEpoch = 50000
  override protected val learningRate = 0.00015
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * value)
}
