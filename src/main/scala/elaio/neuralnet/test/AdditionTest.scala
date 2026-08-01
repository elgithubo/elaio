package elaio.neuralnet.test

object AdditionTest extends MathTest {
  override protected val learningRate = 0.003
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value + 23)
}
