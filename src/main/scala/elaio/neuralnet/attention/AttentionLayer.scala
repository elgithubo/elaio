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
      inputGroups: Array[Array[Double]],
      queryProjection: Array[Array[Double]],
      keyProjection: Array[Array[Double]],
      valueProjection: Array[Array[Double]]
  )
}

// Single-head self-attention core without residuals, normalization, or an output projection.
final class AttentionLayer(val groupWidth: Int, random: Random = new Random) {
  import AttentionLayer.{ForwardPass, Gradients}

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

    for {
      row <- scores.indices
      column <- scores(row).indices
    } scores(row)(column) *= scoreScale

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
    val scoreGradients = Array.ofDim[Double](attentionGradients.length, attentionGradients.length)

    for (row <- attentionGradients.indices) {
      var weightedGradient = 0d
      for (column <- attentionGradients(row).indices)
        weightedGradient += attentionGradients(row)(column) * pass.attentionWeights(row)(column)
      for (column <- attentionGradients(row).indices)
        scoreGradients(row)(column) =
          pass.attentionWeights(row)(column) * (attentionGradients(row)(column) - weightedGradient)
    }

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

  def queryProjectionWeights: Array[Array[Double]] = copyMatrix(queryProjection)
  def keyProjectionWeights: Array[Array[Double]] = copyMatrix(keyProjection)
  def valueProjectionWeights: Array[Array[Double]] = copyMatrix(valueProjection)

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

  private def multiply(left: Array[Array[Double]], right: Array[Array[Double]]): Array[Array[Double]] = {
    require(left.nonEmpty && right.nonEmpty, "matrix operands must not be empty")
    require(left.forall(_.length == right.length), "matrix dimensions do not match")

    val result = Array.ofDim[Double](left.length, right(0).length)
    for {
      row <- left.indices
      column <- right(0).indices
      shared <- right.indices
    } result(row)(column) += left(row)(shared) * right(shared)(column)
    result
  }

  private def transpose(matrix: Array[Array[Double]]): Array[Array[Double]] = {
    val result = Array.ofDim[Double](matrix(0).length, matrix.length)
    for {
      row <- matrix.indices
      column <- matrix(row).indices
    } result(column)(row) = matrix(row)(column)
    result
  }

  private def softmax(values: Array[Double]): Array[Double] = {
    val maximum = values.max
    val exponentials = values.map(value => math.exp(value - maximum))
    val total = exponentials.sum
    exponentials.map(_ / total)
  }

  private def scaleInPlace(matrix: Array[Array[Double]], factor: Double): Unit =
    for {
      row <- matrix.indices
      column <- matrix(row).indices
    } matrix(row)(column) *= factor

  private def add(matrices: Array[Array[Double]]*): Array[Array[Double]] = {
    val result = Array.ofDim[Double](matrices.head.length, matrices.head(0).length)
    for {
      matrix <- matrices
      row <- matrix.indices
      column <- matrix(row).indices
    } result(row)(column) += matrix(row)(column)
    result
  }

  private def squaredSum(matrix: Array[Array[Double]]): Double =
    matrix.iterator.flatMap(_.iterator).map(value => value * value).sum

  private def update(
      weights: Array[Array[Double]],
      gradients: Array[Array[Double]],
      scaledLearningRate: Double
  ): Unit =
    for {
      row <- weights.indices
      column <- weights(row).indices
    } weights(row)(column) -= scaledLearningRate * gradients(row)(column)

  private def copyMatrix(matrix: Array[Array[Double]]): Array[Array[Double]] =
    matrix.map(_.clone())
}
