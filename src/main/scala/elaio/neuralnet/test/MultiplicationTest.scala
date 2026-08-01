package elaio.neuralnet.test

object MultiplicationTest extends MathTest {
  override val tolerance = 5d
  override protected val learningRate = 0.0015
  override protected val clipUntilEpoch = 15000
  protected def targetOf(inputValues: Array[Double]): Array[Double] = inputValues.map(value => value * 3)
}
