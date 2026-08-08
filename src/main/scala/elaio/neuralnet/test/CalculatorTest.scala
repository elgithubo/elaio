package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

// One opcode selects the operation applied to four independent input values.
class CalculatorTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  override protected val dimOuter = 3
  override protected val learningRate = 0.0001d
  override protected val maxUpdateNorm = 1000d
  override protected val epochs = 20000
  override protected val clipUntilEpoch = epochs
  override protected val tolerance = 10d
  override protected val inWidth = 5 // one value reserved for opcode
  override protected val outWidth = 4 // needs to be inWidth - 1
  override protected val trainCount = 400 // needs to be divisible by the number of questions
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

  // opcode and values live in one token, so the whole example is a single row
  private def selectedOperation(tokens: Array[Array[Double]]): Int =
    operations.indices.minBy(operation => math.abs(tokens.head(0) - operations(operation).opcode))

  private def tokensFor(operation: Int, values: Array[Double]): Array[Array[Double]] =
    Array(Array(operations(operation).opcode) ++ values)

  private def randomValues(random: scala.util.Random): Array[Double] =
    Array.fill(outWidth)(randomValue(random))

  override protected def describeInput(tokens: Array[Array[Double]]): String =
    tokens.head.drop(1).map(value => f"$value%.3f").mkString(" | ") +
      "  ->  " + operations(selectedOperation(tokens)).description

  override protected def randomTokens(random: scala.util.Random): Array[Array[Double]] =
    tokensFor(random.nextInt(operations.length), randomValues(random))

  override protected def trainingTokens(random: scala.util.Random): Array[Array[Array[Double]]] = {
    require(trainCount % operations.length == 0, "training examples must divide evenly between operations")
    (for {
      operation <- operations.indices
      _ <- 1 to trainCount / operations.length
    } yield tokensFor(operation, randomValues(random))).toArray
  }

  // ask each operation equally often and grouped, so the log reads one operation at a time
  override protected def checkTokens(random: scala.util.Random): Seq[Array[Array[Double]]] =
    for {
      operation <- operations.indices
      _ <- 1 to numberOfQuestions / operations.length
    } yield tokensFor(operation, randomValues(random))

  protected def targetOf(tokens: Array[Array[Double]]): Array[Double] = {
    val operation = operations(selectedOperation(tokens)).calculate
    tokens.head.drop(1).map(operation)
  }
}
