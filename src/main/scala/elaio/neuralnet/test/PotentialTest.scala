package elaio.neuralnet.test

object PotentialTest extends MathTest {
  override val tolerance = 5d
  override val epochs = 20000
  override val clipUntilEpoch = 20000
  override protected val learningRate = 0.00015
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * value)
}
