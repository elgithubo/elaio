package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

// One net for four operations. The first input carries the value, the remaining four are a
// one-hot opcode selecting which operation to apply. The net has to form
// out = sum over k of opcode_k * f_k(value), so a gate times a data path - a product between
// two activations, which is exactly what the squaring units provide.
final class OpcodeTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val learningRate = 0.0001d
  override protected val maxUpdateNorm = 1000d
  override protected val clipUntilEpoch = 15000
  override protected val tolerance = 5d
  // four operations need more than five questions to all get asked
  override protected val numberOfQuestions = 20

  // the opcode must match the value's magnitude, otherwise the averaged fan-in drowns it
  private val opcodeLevel = 1000d
  private val operations: Array[Double => Double] =
    Array(_ + 23d, _ - 13d, _ * 3d, _ / 5d)

  override protected def randomInput(random: scala.util.Random): Array[Double] = {
    val input = Array.fill(5)(0d)
    input(0) = random.nextDouble() * 2000d - 1000d
    input(1 + random.nextInt(operations.length)) = opcodeLevel
    input
  }

  // every output carries the same result, so a run scores 5 of 5 per question
  protected def targetOf(inputValues: Array[Double]): Array[Double] = {
    val selected = inputValues.indexWhere(_ == opcodeLevel, 1) - 1
    Array.fill(5)(operations(selected)(inputValues(0)))
  }
}
