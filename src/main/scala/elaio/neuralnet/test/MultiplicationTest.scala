package elaio.neuralnet.test

object MultiplicationTest extends MathTest {
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * 3)
}
