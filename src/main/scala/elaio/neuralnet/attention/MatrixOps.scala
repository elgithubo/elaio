package elaio.neuralnet.attention

// Indexed loops avoid allocation in attention's per-example matrix operations.
private[attention] object MatrixOps {
  def multiply(left: Array[Array[Double]], right: Array[Array[Double]]): Array[Array[Double]] = {
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

  def transpose(matrix: Array[Array[Double]]): Array[Array[Double]] = {
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

  def softmax(values: Array[Double]): Array[Double] = {
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

  def scaleInPlace(matrix: Array[Array[Double]], factor: Double): Unit = {
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

  def squaredSum(matrix: Array[Array[Double]]): Double = {
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

  def update(
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

  def copyMatrix(matrix: Array[Array[Double]]): Array[Array[Double]] =
    matrix.map(_.clone())
}
