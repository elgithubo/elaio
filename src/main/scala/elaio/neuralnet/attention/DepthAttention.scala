package elaio.neuralnet.attention

import elaio.neuralnet.attention.AttentionLayer.ForwardPass
import elaio.neuralnet.processing.NeuronGroup
import elaio.neuralnet.units.InputNeuron

// Attention across the depth of a graph: one depth group is one token, the neurons inside it
// are the channels. The result is written back as an additive context on the pre-activation,
// so a second forward pass sees the refined values. Knows nothing about how the graph was built.
final class DepthAttention(groups: Vector[NeuronGroup], contextScale: Double = 0.1d) {
  require(groups.nonEmpty, "depth attention needs at least one group")

  private val layer = new AttentionLayer(groups.map(_.neurons.length).max)

  // one plain pass to read from, then one more with the attention context in place
  def refine(forward: () => Unit): ForwardPass = {
    clearContexts()
    forward()
    val pass = layer.forward(groupValues())
    applyContexts(pass)
    try forward()
    finally clearContexts()
    pass
  }

  // The gradient into the first graph pass is intentionally truncated for now.
  def applyGradients(pass: ForwardPass, learningRate: Double, maxGradientNorm: Double): Unit = {
    // delta is -dL/dz and the context is added to z, so the loss gradient carries the minus
    val outputGradients = Array.ofDim[Double](groups.length, layer.groupWidth)
    for {
      groupIndex <- groups.indices
      neuronIndex <- groups(groupIndex).neurons.indices
      neuron = groups(groupIndex).neurons(neuronIndex)
      if !neuron.isInstanceOf[InputNeuron]
    } outputGradients(groupIndex)(neuronIndex) = -neuron.delta * contextScale

    layer.applyGradients(layer.backward(pass, outputGradients), learningRate, maxGradientNorm)
  }

  // every group normalized to unit RMS, padded to the widest group
  private def groupValues(): Array[Array[Double]] =
    groups.map { group =>
      require(group.neurons.forall(_.value.isFinite), "attention input values must be finite")
      val row = Array.ofDim[Double](layer.groupWidth)
      val meanSquare =
        group.neurons.iterator.map(neuron => neuron.value * neuron.value).sum / group.neurons.length
      val rms = math.sqrt(meanSquare + 1e-8)
      for (neuronIndex <- group.neurons.indices)
        row(neuronIndex) = group.neurons(neuronIndex).value / rms
      row
    }.toArray

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
