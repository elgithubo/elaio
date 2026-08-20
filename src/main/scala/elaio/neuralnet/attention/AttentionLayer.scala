package elaio.neuralnet.attention

import scala.util.Random

object AttentionLayer {
  final case class ForwardPass private[attention] (
      inputGroups: Array[Array[Double]],
      queries: Array[Array[Double]],
      keys: Array[Array[Double]],
      values: Array[Array[Double]],
      attentionWeights: Array[Array[Double]],
      outputGroups: Array[Array[Double]]
  )

  final case class Gradients(
      inputGradients: Array[Array[Double]],
      queryProjection: Array[Array[Double]],
      keyProjection: Array[Array[Double]],
      valueProjection: Array[Array[Double]]
  )
}

// Single-head self-attention core without residuals, normalization, or an output projection.
final class AttentionLayer(val groupWidth: Int, random: Random = new Random) {
  import AttentionLayer.{ForwardPass, Gradients}
  import MatrixOps.*

  require(groupWidth > 0, "attention group width must be positive")

  private val queryProjection = initializedProjection()
  private val keyProjection = initializedProjection()
  private val valueProjection = initializedProjection()
  private val scoreScale = 1d / math.sqrt(groupWidth.toDouble)

  def forward(inputGroups: Array[Array[Double]]): ForwardPass = {
    requireGroups(inputGroups, "input groups")

    val inputs = copyMatrix(inputGroups)
    val queries = multiply(inputs, queryProjection)
    val keys = multiply(inputs, keyProjection)
    val values = multiply(inputs, valueProjection)
    val scores = multiply(queries, transpose(keys))

    scaleInPlace(scores, scoreScale)

    val attentionWeights = scores.map(softmax)
    val outputGroups = multiply(attentionWeights, values)
    ForwardPass(inputs, queries, keys, values, attentionWeights, outputGroups)
  }

  def backward(pass: ForwardPass, outputGradients: Array[Array[Double]]): Gradients = {
    requireGroups(outputGradients, "output gradients")
    require(
      outputGradients.length == pass.outputGroups.length,
      "output gradients must contain one row per output group"
    )

    val attentionGradients = multiply(outputGradients, transpose(pass.values))
    val valueGradients = multiply(transpose(pass.attentionWeights), outputGradients)
    val scoreGradients = calculateScoreGradients(attentionGradients, pass.attentionWeights)

    val queryGradients = multiply(scoreGradients, pass.keys)
    val keyGradients = multiply(transpose(scoreGradients), pass.queries)
    scaleInPlace(queryGradients, scoreScale)
    scaleInPlace(keyGradients, scoreScale)

    val inputTranspose = transpose(pass.inputGroups)
    val queryProjectionGradients = multiply(inputTranspose, queryGradients)
    val keyProjectionGradients = multiply(inputTranspose, keyGradients)
    val valueProjectionGradients = multiply(inputTranspose, valueGradients)

    val inputGradients = add(
      multiply(queryGradients, transpose(queryProjection)),
      multiply(keyGradients, transpose(keyProjection)),
      multiply(valueGradients, transpose(valueProjection))
    )

    Gradients(
      inputGradients,
      queryProjectionGradients,
      keyProjectionGradients,
      valueProjectionGradients
    )
  }

  def applyGradients(
      gradients: Gradients,
      learningRate: Double,
      maxGradientNorm: Double = Double.PositiveInfinity
  ): Unit = {
    require(learningRate > 0d && learningRate.isFinite, "learning rate must be positive and finite")
    require(maxGradientNorm > 0d, "maximum gradient norm must be positive")
    requireProjection(gradients.queryProjection, "query projection gradients")
    requireProjection(gradients.keyProjection, "key projection gradients")
    requireProjection(gradients.valueProjection, "value projection gradients")

    val norm = math.sqrt(
      squaredSum(gradients.queryProjection) +
        squaredSum(gradients.keyProjection) +
        squaredSum(gradients.valueProjection)
    )
    require(norm.isFinite, "attention gradients must be finite")

    val scale = if (norm > maxGradientNorm) maxGradientNorm / norm else 1d
    update(queryProjection, gradients.queryProjection, learningRate * scale)
    update(keyProjection, gradients.keyProjection, learningRate * scale)
    update(valueProjection, gradients.valueProjection, learningRate * scale)
  }

  private def calculateScoreGradients(
      attentionGradients: Array[Array[Double]],
      attentionWeights: Array[Array[Double]]
  ): Array[Array[Double]] = {
    val scoreGradients = Array.ofDim[Double](attentionGradients.length, attentionGradients.length)
    var row = 0
    while (row < attentionGradients.length) {
      val attentionGradientRow = attentionGradients(row)
      val attentionWeightRow = attentionWeights(row)
      val scoreGradientRow = scoreGradients(row)
      var weightedGradient = 0d
      var column = 0
      while (column < attentionGradientRow.length) {
        weightedGradient += attentionGradientRow(column) * attentionWeightRow(column)
        column += 1
      }
      column = 0
      while (column < attentionGradientRow.length) {
        scoreGradientRow(column) = attentionWeightRow(column) * (attentionGradientRow(column) - weightedGradient)
        column += 1
      }
      row += 1
    }
    scoreGradients
  }

  private def initializedProjection(): Array[Array[Double]] = {
    val deviation = 1d / math.sqrt(groupWidth.toDouble)
    Array.fill(groupWidth, groupWidth)(random.nextGaussian() * deviation)
  }

  private def requireGroups(groups: Array[Array[Double]], name: String): Unit = {
    require(groups.nonEmpty, s"$name must not be empty")
    require(groups.forall(_.length == groupWidth), s"every row in $name must have width $groupWidth")
  }

  private def requireProjection(projection: Array[Array[Double]], name: String): Unit =
    require(
      projection.length == groupWidth && projection.forall(_.length == groupWidth),
      s"$name must have shape ${groupWidth}x$groupWidth"
    )
}
