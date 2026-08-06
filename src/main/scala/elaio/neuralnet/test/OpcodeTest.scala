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
  // the number of questions must be divisible by the number of operations to keep them grouped
  override protected val numberOfQuestions = 20

  // the opcode must match the value's magnitude, otherwise the averaged fan-in drowns it
  private val opcodeLevel = 1000d
  private val operations: Array[(String, Double => Double)] =
    Array(
      "x + 23" -> ((x: Double) => x + 23d),
      "x - 13" -> ((x: Double) => x - 13d),
      "x * 3"  -> ((x: Double) => x * 3d),
      "x / 5"  -> ((x: Double) => x / 5d)
    )

  // the opcode channels carry no data, so show the value and the operation it selects
  override protected def describeInput(inputValues: Array[Double]): String =
    f"${inputValues(0)}%.3f  ->  ${operations(selectedOperation(inputValues))._1}"

  private def selectedOperation(inputValues: Array[Double]): Int =
    inputValues.indices.drop(1).maxBy(inputValues) - 1

  private def inputFor(operation: Int, value: Double): Array[Double] = {
    val input = Array.fill(5)(0d)
    input(0) = value
    input(1 + operation) = opcodeLevel
    input
  }

  private def randomValue(random: scala.util.Random): Double = random.nextDouble() * 2000d - 1000d

  override protected def randomInput(random: scala.util.Random): Array[Double] =
    inputFor(random.nextInt(operations.length), randomValue(random))

  // ask each operation equally often and grouped, so the log reads one operation at a time
  override protected def checkInputs(random: scala.util.Random): Seq[Array[Double]] =
    for {
      operation <- operations.indices
      _ <- 1 to numberOfQuestions / operations.length
    } yield inputFor(operation, randomValue(random))

  // every output carries the same result, so a run scores 5 of 5 per question.
  // the opcode is read as the largest of its channels rather than by equality, so a value that
  // has been through a file or a computation still selects instead of falling through to -1.
  protected def targetOf(inputValues: Array[Double]): Array[Double] = {
    val selected = selectedOperation(inputValues)
    Array.fill(5)(operations(selected)._2(inputValues(0)))
  }
}
