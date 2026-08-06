package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

// One opcode selects the operation applied to four independent input values.
final class CalculatorTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val dimOuter = 3
  override protected val learningRate = 0.0001d
  override protected val maxUpdateNorm = 1000d
  override protected val epochs = 30000
  override protected val clipUntilEpoch = 30000
  override protected val tolerance = 10d
  override protected val outWidth = 4
  override protected val trainCount = 400 // needs to be divisible by the number of operations
  override protected val numberOfQuestions = 20

  private final case class Operation(opcode: Double, description: String, calculate: Double => Double)

  private val operations =
    // operation code, operation description, operation calculation
    Array(
      Operation(-1000d, "x + 23", (x: Double) => x + 23d),
      Operation( -333d, "x - 13", (x: Double) => x - 13d),
      Operation(  333d, "x * 3",  (x: Double) => x * 3d ),
      Operation( 1000d, "x / 5",  (x: Double) => x / 5d )
    )

  private def selectedOperation(inputValues: Array[Double]): Int =
    operations.indices.minBy(operation => math.abs(inputValues(0) - operations(operation).opcode))

  private def inputFor(operation: Int, values: Array[Double]): Array[Double] =
    Array(operations(operation).opcode) ++ values

  private def randomValues(random: scala.util.Random): Array[Double] =
    Array.fill(outWidth)(randomValue(random))

  override protected def describeInput(inputValues: Array[Double]): String =
    inputValues.drop(1).map(value => f"$value%.3f").mkString(" | ") +
      "  ->  " + operations(selectedOperation(inputValues)).description

  override protected def randomInput(random: scala.util.Random): Array[Double] =
    inputFor(random.nextInt(operations.length), randomValues(random))

  override protected def trainingInputs(random: scala.util.Random): Array[Array[Double]] = {
    require(trainCount % operations.length == 0, "training examples must divide evenly between operations")
    (for {
      operation <- operations.indices
      _ <- 1 to trainCount / operations.length
    } yield inputFor(operation, randomValues(random))).toArray
  }

  // ask each operation equally often and grouped, so the log reads one operation at a time
  override protected def checkInputs(random: scala.util.Random): Seq[Array[Double]] =
    for {
      operation <- operations.indices
      _ <- 1 to numberOfQuestions / operations.length
    } yield inputFor(operation, randomValues(random))

  protected def targetOf(inputValues: Array[Double]): Array[Double] = {
    val operation = operations(selectedOperation(inputValues)).calculate
    inputValues.drop(1).map(operation)
  }
}
