package elaio.neuralnet.test

object PotentialTest extends MathTest {
  override val tolerance = 2d
  override val epochs = 20000
  override val clipUntilEpoch = 20000
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * value)
}
