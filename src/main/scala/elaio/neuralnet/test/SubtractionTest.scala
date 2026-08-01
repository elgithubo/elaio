package elaio.neuralnet.test

object SubtractionTest extends MathTest {
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value - 13)
}
