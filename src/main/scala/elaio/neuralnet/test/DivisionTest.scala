package elaio.neuralnet.test

object DivisionTest extends MathTest {
  override val tolerance = 0.1d
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value / 5)
}
