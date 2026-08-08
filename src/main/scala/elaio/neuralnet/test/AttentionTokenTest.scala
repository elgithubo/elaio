package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction

// Learns to retrieve the random value associated with a queried key.
final class AttentionTokenTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  private val keyCount = 3
  private val tokenWidth = keyCount + 2 // query marker, one-hot key, value
  private val tokenCount = keyCount + 1 // one memory per key and one query

  override protected val attentionEnabled: Boolean = true

  override protected val dimOuter = 2
  override protected val inWidth = tokenCount * tokenWidth
  override protected val outWidth = 1
  override protected val inputMinimum = -1d
  override protected val inputMaximum = 1d
  override protected val trainCount = 300
  override protected val numberOfQuestions = 12
  override protected val epochs = 10000
  override protected val clipUntilEpoch = epochs
  override protected val learningRate = 0.005d
  override protected val maxUpdateNorm = 100d
  override protected val tolerance = 0.1d

  private def keyVector(key: Int): Array[Double] =
    Array.tabulate(keyCount)(index => if (index == key) 1d else 0d)

  private def memoryToken(key: Int, value: Double): Array[Double] =
    Array(0d) ++ keyVector(key) ++ Array(value)

  private def queryToken(key: Int): Array[Double] =
    Array(1d) ++ keyVector(key) ++ Array(0d)

  private def randomTokens(random: scala.util.Random, queryKey: Int): Array[Array[Double]] = {
    val memories = random.shuffle(
      (0 until keyCount).map(key => memoryToken(key, randomValue(random)))
    ).toArray
    memories :+ queryToken(queryKey)
  }

  private def tokensOf(inputValues: Array[Double]): Array[Array[Double]] =
    inputValues.grouped(tokenWidth).map(_.toArray).toArray

  private def keyOf(token: Array[Double]): Int =
    (0 until keyCount).maxBy(index => token(index + 1))

  override protected def randomInput(random: scala.util.Random): Array[Double] =
    randomTokens(random, random.nextInt(keyCount)).flatten

  override protected def trainingInputs(random: scala.util.Random): Array[Array[Double]] = {
    require(trainCount % keyCount == 0, "training examples must divide evenly between keys")
    (for {
      queryKey <- 0 until keyCount
      _ <- 1 to trainCount / keyCount
    } yield randomTokens(random, queryKey).flatten).toArray
  }

  override protected def checkInputs(random: scala.util.Random): Seq[Array[Double]] = {
    require(numberOfQuestions % keyCount == 0, "questions must divide evenly between keys")
    for {
      queryKey <- 0 until keyCount
      _ <- 1 to numberOfQuestions / keyCount
    } yield randomTokens(random, queryKey).flatten
  }

  override protected def describeInput(inputValues: Array[Double]): String = {
    val tokens = tokensOf(inputValues)
    val memories = tokens.filter(_(0) == 0d).map { token =>
      s"${('A' + keyOf(token)).toChar} -> ${token.last}"
    }
    val query = tokens.find(_(0) == 1d).get
    memories.mkString(" | ") + s" | query ${('A' + keyOf(query)).toChar}"
  }

  protected def targetOf(inputValues: Array[Double]): Array[Double] = {
    val tokens = tokensOf(inputValues)
    val query = tokens.find(_(0) == 1d).get
    val queryKey = keyOf(query)
    val memory = tokens.find(token => token(0) == 0d && keyOf(token) == queryKey).get
    Array(memory.last)
  }
}
