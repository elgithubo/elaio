package elaio.neuralnet.processing

import elaio.neuralnet.units.OutputNeuron

object Backpropagation {
  // processing neuron graph's reverse order.
  // note that a forward pass must have run.
  //
  // delta is -dL/dz for L = 0.5*(target - value)^2
  //
  // here comes the math from the ai
  // collectInConnections averages instead of summing, so the same 1/N appears here:
  //   delta_j = f'(z_j) * sum_k (delta_k * w_jk / N_k)   <- N of the target k
  //   dw_ij   = delta_j * a_i / N_j                      <- N of the owner j
  // maxUpdateNorm caps the length of the whole update vector, leaving its direction
  // alone since it is the extreme steps that blow the net
  //
  // returns the raw length of the update before the cap, so the caller can see whether the cap
  // binds - NaN when there is no cap and the length was never computed
  def run(order: GraphTraversal.ReverseOrder, learningRate: Double,
          maxUpdateNorm: Double = Double.PositiveInfinity): Double = {
    for (output <- order.outputs) {
      val outputNeuron = output.asInstanceOf[OutputNeuron]
      outputNeuron.delta =
        (outputNeuron.target - outputNeuron.value) * outputNeuron.activationDerivative(outputNeuron.preActivation)
    }

    for (neuron <- order.sequence.iterator if !order.outputs.contains(neuron))
      neuron.delta =
        neuron.connectionsOut.foldLeft(0d) { (sum, connection) =>
          val targetNeuron = connection.neuronTarget
          if (order.reachable.contains(targetNeuron))
            sum + connection.weight * targetNeuron.delta / targetNeuron.connectionsIn.length
          else sum
        } * neuron.activationDerivative(neuron.preActivation) // outgoing sum * activation derivative

    if (order.hasSharedParameters) {
      // Shared cells receive one summed, clipped update.
      order.connectionWeights.foreach(_.accumulatedGradient = 0d)
      order.neuronBiases.foreach(_.accumulatedGradient = 0d)
      for (neuron <- order.sequence.reverseIterator) {
        val fanIn = neuron.connectionsIn.length
        for (connectionIn <- neuron.connectionsIn)
          connectionIn.weightCell.accumulatedGradient +=
            neuron.delta * connectionIn.neuronSource.value / fanIn
        if (fanIn > 0)
          neuron.biasCell.accumulatedGradient += neuron.delta
      }

      val weightGradientSquares =
        order.connectionWeights.iterator.map(weight => weight.accumulatedGradient * weight.accumulatedGradient).sum
      val biasGradientSquares =
        order.neuronBiases.iterator.map(bias => bias.accumulatedGradient * bias.accumulatedGradient).sum
      val norm = math.sqrt(weightGradientSquares + biasGradientSquares)
      val scale = if (norm > maxUpdateNorm) maxUpdateNorm / norm else 1d

      for (weight <- order.connectionWeights) {
        weight.value += learningRate * scale * weight.accumulatedGradient
        weight.accumulatedGradient = 0d
      }
      for (bias <- order.neuronBiases) {
        bias.value += learningRate * scale * bias.accumulatedGradient
        bias.accumulatedGradient = 0d
      }
      norm
    } else {
      // without a cap the length is left uncomputed - it would cost a full extra pass for nothing
      val norm =
        if (maxUpdateNorm.isPosInfinity) Double.NaN
        else {
          var sumSquares = 0d
          for (neuron <- order.sequence.reverseIterator) {
            val fanIn = neuron.connectionsIn.length
            for (connectionIn <- neuron.connectionsIn) {
              val gradient = neuron.delta * connectionIn.neuronSource.value / fanIn
              sumSquares += gradient * gradient
            }
            if (fanIn > 0) sumSquares += neuron.delta * neuron.delta
          }
          math.sqrt(sumSquares)
        }
      // a NaN norm compares false, so an uncapped run keeps its scale of 1
      val scale = if (norm > maxUpdateNorm) maxUpdateNorm / norm else 1d

      for (neuron <- order.sequence.reverseIterator) {
        val fanIn = neuron.connectionsIn.length
        for (connectionIn <- neuron.connectionsIn)
          connectionIn.weight +=
            learningRate * scale * neuron.delta * connectionIn.neuronSource.value / fanIn
        if (fanIn > 0) // skip input neurons
          neuron.bias += learningRate * scale * neuron.delta
      }
      norm
    }
  }
}
