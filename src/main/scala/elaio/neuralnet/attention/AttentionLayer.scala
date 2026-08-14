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

  // Attention runs per example, so its hot matrix loops use indexed access to avoid allocations.
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
    val scoreGradients = Array.ofDim[Double](attentionGradients.length, attentionGradients.length)

    var row = 0
    while (row < attentionGradients.length) {
      val attentionGradientRow = attentionGradients(row)
      val attentionWeightRow = pass.attentionWeights(row)
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

    val queryGradients = multiply(scoreGradients, pass.keys)
    val keyGradients = multiply(transpose(scoreGradients), pass.queries)
    scaleInPlace(queryGradients, scoreScale)
    scaleInPlace(keyGradients, scoreScale)

    val inputTranspose = transpose(pass.inputGroups)
    val queryProjectionGradients = multiply(inputTranspose, queryGradients)
    val keyProjectionGradients = multiply(inputTranspose, keyGradients)
    val valueProjectionGradients = multiply(inputTranspose, valueGradients)

    Gradients(
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
    var row = 0
    while (row < left.length) {
      val leftRow = left(row)
      val resultRow = result(row)
      var shared = 0
      while (shared < right.length) {
        val leftValue = leftRow(shared)
        val rightRow = right(shared)
        var column = 0
        while (column < resultRow.length) {
          resultRow(column) += leftValue * rightRow(column)
          column += 1
        }
        shared += 1
      }
      row += 1
    }
    result
  }

  private def transpose(matrix: Array[Array[Double]]): Array[Array[Double]] = {
    val result = Array.ofDim[Double](matrix(0).length, matrix.length)
    var row = 0
    while (row < matrix.length) {
      val matrixRow = matrix(row)
      var column = 0
      while (column < matrixRow.length) {
        result(column)(row) = matrixRow(column)
        column += 1
      }
      row += 1
    }
    result
  }

  private def softmax(values: Array[Double]): Array[Double] = {
    var maximum = values(0)
    var index = 1
    while (index < values.length) {
      maximum = math.max(maximum, values(index))
      index += 1
    }

    val exponentials = Array.ofDim[Double](values.length)
    var total = 0d
    index = 0
    while (index < values.length) {
      val exponential = math.exp(values(index) - maximum)
      exponentials(index) = exponential
      total += exponential
      index += 1
    }
    index = 0
    while (index < exponentials.length) {
      exponentials(index) /= total
      index += 1
    }
    exponentials
  }

  private def scaleInPlace(matrix: Array[Array[Double]], factor: Double): Unit = {
    var row = 0
    while (row < matrix.length) {
      val matrixRow = matrix(row)
      var column = 0
      while (column < matrixRow.length) {
        matrixRow(column) *= factor
        column += 1
      }
      row += 1
    }
  }

  private def add(matrices: Array[Array[Double]]*): Array[Array[Double]] = {
    val result = Array.ofDim[Double](matrices.head.length, matrices.head(0).length)
    for {
      matrix <- matrices
      row <- matrix.indices
      column <- matrix(row).indices
    } result(row)(column) += matrix(row)(column)
    result
  }

  private def squaredSum(matrix: Array[Array[Double]]): Double = {
    var result = 0d
    var row = 0
    while (row < matrix.length) {
      val matrixRow = matrix(row)
      var column = 0
      while (column < matrixRow.length) {
        val value = matrixRow(column)
        result += value * value
        column += 1
      }
      row += 1
    }
    result
  }

  private def update(
      weights: Array[Array[Double]],
      gradients: Array[Array[Double]],
      scaledLearningRate: Double
  ): Unit = {
    var row = 0
    while (row < weights.length) {
      val weightRow = weights(row)
      val gradientRow = gradients(row)
      var column = 0
      while (column < weightRow.length) {
        weightRow(column) -= scaledLearningRate * gradientRow(column)
        column += 1
      }
      row += 1
    }
  }

  private def copyMatrix(matrix: Array[Array[Double]]): Array[Array[Double]] =
    matrix.map(_.clone())
}
