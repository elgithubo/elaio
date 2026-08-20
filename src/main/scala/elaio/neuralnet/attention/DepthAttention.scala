package elaio.neuralnet.attention

import elaio.neuralnet.attention.AttentionLayer.{ForwardPass, Gradients}
import elaio.neuralnet.processing.{Backpropagation, GraphTraversal}
import elaio.neuralnet.processing.GraphTraversal.ReverseOrder
import elaio.neuralnet.units.{InputNeuron, Neuron}

// Attention across the depth of a graph: one depth group is one token, the neurons inside it
// are the channels. The result is written back as an additive context on the pre-activation,
// so a second forward pass sees the refined values. Knows nothing about how the graph was built.
final class DepthAttention(boundOrder: ReverseOrder, contextScale: Double = 0.1d) {
  private val normalizationEpsilon = 1e-8
  private val groups = GraphTraversal.depthGroups(boundOrder)
  require(groups.nonEmpty, "depth attention needs at least one group")

  private val layer = new AttentionLayer(groups.map(_.neurons.length).max)

  // One plain pass supplies attention, then a refined pass produces the network output.
  def refine(currentOrder: ReverseOrder, forward: () => Unit): ForwardPass = {
    requireBoundOrder(currentOrder)
    clearContexts()
    forward()
    val pass = layer.forward(normalizedGroupValues())
    try {
      applyContexts(pass)
      forward()
    } finally clearContexts()
    pass
  }

  // Reads the refined graph deltas before its state is replaced by first-pass recomputation.
  def gradientsFromRefinedDeltas(pass: ForwardPass): Gradients =
    layer.backward(pass, attentionOutputGradients())

  // Restores the plain graph state without changing parameters or creating another attention pass.
  def recomputeFirstPass(currentOrder: ReverseOrder, forward: () => Unit): Unit = {
    requireBoundOrder(currentOrder)
    clearContexts()
    forward()
  }

  // Converts gradients through RMS normalization and seeds direct first-pass neuron deltas.
  def resetAndSeedFirstPassDeltas(gradients: Gradients): Unit = {
    requireInputGradientShape(gradients.inputGradients)
    Backpropagation.clearDeltas(boundOrder)
    for (groupIndex <- groups.indices)
      seedGroupDeltas(groups(groupIndex).neurons, gradients.inputGradients(groupIndex))
  }

  def applyGradients(gradients: Gradients, learningRate: Double, maxGradientNorm: Double): Unit =
    layer.applyGradients(gradients, learningRate, maxGradientNorm)

  private def attentionOutputGradients(): Array[Array[Double]] = {
    val result = Array.ofDim[Double](groups.length, layer.groupWidth)
    for {
      groupIndex <- groups.indices
      neuronIndex <- groups(groupIndex).neurons.indices
      neuron = groups(groupIndex).neurons(neuronIndex)
      if !neuron.isInstanceOf[InputNeuron]
    } result(groupIndex)(neuronIndex) = -neuron.delta * contextScale
    result
  }

  // Every group is normalized to unit RMS and padded to the widest group.
  private def normalizedGroupValues(): Array[Array[Double]] =
    groups.map { group =>
      val rms = groupRms(group.neurons)
      val row = Array.ofDim[Double](layer.groupWidth)
      for (neuronIndex <- group.neurons.indices)
        row(neuronIndex) = group.neurons(neuronIndex).value / rms
      row
    }.toArray

  private def seedGroupDeltas(neurons: Vector[Neuron], normalizedGradients: Array[Double]): Unit = {
    val rms = groupRms(neurons)
    var gradientValueProduct = 0d
    var index = 0
    while (index < neurons.length) {
      gradientValueProduct += normalizedGradients(index) * neurons(index).value
      index += 1
    }

    val normalizationDivisor = neurons.length * rms * rms * rms
    index = 0
    while (index < neurons.length) {
      val neuron = neurons(index)
      if (!neuron.isInstanceOf[InputNeuron]) {
        val valueGradient =
          normalizedGradients(index) / rms - neuron.value * gradientValueProduct / normalizationDivisor
        neuron.delta = -valueGradient * neuron.activationDerivative(neuron.preActivation)
      }
      index += 1
    }
  }

  private def groupRms(neurons: Vector[Neuron]): Double = {
    require(neurons.forall(_.value.isFinite), "attention input values must be finite")
    val meanSquare = neurons.iterator.map(neuron => neuron.value * neuron.value).sum / neurons.length
    math.sqrt(meanSquare + normalizationEpsilon)
  }

  private def requireInputGradientShape(inputGradients: Array[Array[Double]]): Unit = {
    require(inputGradients.length == groups.length, "attention input gradients need one row per depth group")
    require(
      inputGradients.forall(_.length == layer.groupWidth),
      s"attention input gradient rows must have width ${layer.groupWidth}"
    )
  }

  private def requireBoundOrder(currentOrder: ReverseOrder): Unit =
    require(currentOrder eq boundOrder, "depth attention belongs to a different graph")

  private def applyContexts(pass: ForwardPass): Unit =
    for {
      groupIndex <- groups.indices
      neuronIndex <- groups(groupIndex).neurons.indices
      neuron = groups(groupIndex).neurons(neuronIndex)
      if !neuron.isInstanceOf[InputNeuron]
    } {
      val context = pass.outputGroups(groupIndex)(neuronIndex) * contextScale
      require(context.isFinite, "attention context values must be finite")
      neuron.attentionContext_=(context)
    }

  private def clearContexts(): Unit =
    for {
      group <- groups
      neuron <- group.neurons
    } neuron.attentionContext_=(0d)
}
