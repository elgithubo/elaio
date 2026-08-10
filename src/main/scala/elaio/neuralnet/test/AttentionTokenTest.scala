package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction
import elaio.neuralnet.trace.NetTrace

// Learns to retrieve the random value associated with a queried key.
//
// data structure:
// Each example contains four tokens with five values each:
// Token layout: [query marker, one-hot key A/B/C, value]; the target is the queried value.
// query marker is 0 or one depending on whether it is teaching or asking.
// [0, 1, 0, 0,  0.42]  |  A → 0.42
// [0, 0, 0, 1, -0.70]  |  C → -0.70
// [0, 0, 1, 0,  0.15]  |  B → 0.15
// [1, 0, 0, 1,  0.00]  |  query C
// => target: [-0.70]
class AttentionTokenTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends MathTest {
  private val keyCount = 3
  private val keyTokenWidth = keyCount + 2 // query marker, one-hot key, value
  private val keyTokenCount = keyCount + 1 // one memory per key and one query

  override protected val attentionEnabled: Boolean = true

  override protected def tokenWidth: Int = keyTokenWidth
  override protected val dimOuter = 2
  override protected val inWidth = keyTokenCount * keyTokenWidth
  override protected val outWidth = 1
  override protected val inputMinimum = -1d
  override protected val inputMaximum = 1d
  override protected val trainCount = 300
  override protected val numberOfQuestions = 300
  override protected val epochs = 20000
  override protected val clipUntilEpoch = 100
  override protected val learningRate = 0.005d
  override protected val maxUpdateNorm = 100d
  override protected val tolerance = 0.1d

  private def keyVector(key: Int): Array[Double] =
    Array.tabulate(keyCount)(index => if (index == key) 1d else 0d)

  private def memoryToken(key: Int, value: Double): Array[Double] =
    Array(0d) ++ keyVector(key) ++ Array(value)

  private def queryToken(key: Int): Array[Double] =
    Array(1d) ++ keyVector(key) ++ Array(0d)

  private def tokensFor(random: scala.util.Random, queryKey: Int): Array[Array[Double]] = {
    val memories = random.shuffle(
      (0 until keyCount).map(key => memoryToken(key, randomValue(random)))
    ).toArray
    memories :+ queryToken(queryKey)
  }

  private def keyOf(token: Array[Double]): Int =
    (0 until keyCount).maxBy(index => token(index + 1))

  override protected def traceAction(): Unit =
    NetTrace.WriteMessage("testing: attention with a token matrix")

  override protected def randomTokens(random: scala.util.Random): TokenMatrix =
    tokensFor(random, random.nextInt(keyCount))

  override protected def trainingTokens(random: scala.util.Random): Array[TokenMatrix] = {
    require(trainCount % keyCount == 0, "training examples must divide evenly between keys")
    (for {
      queryKey <- 0 until keyCount
      _ <- 1 to trainCount / keyCount
    } yield tokensFor(random, queryKey)).toArray
  }

  override protected def checkTokens(random: scala.util.Random): Seq[TokenMatrix] = {
    require(numberOfQuestions % keyCount == 0, "questions must divide evenly between keys")
    for {
      queryKey <- 0 until keyCount
      _ <- 1 to numberOfQuestions / keyCount
    } yield tokensFor(random, queryKey)
  }

  override protected def describeInput(tokens: TokenMatrix): String = {
    val memories = tokens.filter(_(0) == 0d).map { token =>
      s"${('A' + keyOf(token)).toChar} -> ${token.last}"
    }
    val query = tokens.find(_(0) == 1d).get
    memories.mkString(" | ") + s" | query ${('A' + keyOf(query)).toChar}"
  }

  protected def targetOf(tokens: TokenMatrix): Array[Double] = {
    val query = tokens.find(_(0) == 1d).get
    val queryKey = keyOf(query)
    val memory = tokens.find(token => token(0) == 0d && keyOf(token) == queryKey).get
    Array(memory.last)
  }
}
