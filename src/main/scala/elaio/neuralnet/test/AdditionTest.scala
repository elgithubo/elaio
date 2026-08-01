package elaio.neuralnet.test

object AdditionTest extends MathTest {
  override val epochs = 15000
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value + 23)
}
