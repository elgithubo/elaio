package elaio.neuralnet.test

object DivisionTest extends MathTest {
  override val tolerance = 0.25d
  override protected val learningRate = 0.015
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value / 5)
}
