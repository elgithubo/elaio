package elaio.neuralnet.processing

import elaio.neuralnet.units.{Neuron, OutputNeuron}

object Backpropagation {
  // processing neuron graph's reverse order.
  // note that a forward pass must have run.
  //
  // here comes the math from the ai
  // collectInConnections averages instead of summing, so the same 1/N appears here:
  //   delta_j = f'(z_j) * sum_k (delta_k * w_jk / N_k)   <- N of the target k
  //   dw_ij   = delta_j * a_i / N_j                      <- N of the owner j

  // delta is -dL/dz for L = 0.5*(target - value)^2
  def calculateOutputDeltas(order: GraphTraversal.ReverseOrder): Unit =
    for (output <- order.outputs) {
      val outputNeuron = output.asInstanceOf[OutputNeuron]
      outputNeuron.delta =
        (outputNeuron.target - outputNeuron.value) * outputNeuron.activationDerivative(outputNeuron.preActivation)
    }

  // Deltas for one reverse-topological slice of the graph. The neurons the slice feeds into must
  // already carry their deltas. A slice writes only its own neurons, so disjoint slices are safe
  // to run concurrently.
  def calculateDeltas(sequence: Vector[Neuron], reachable: Set[Neuron], skip: Set[Neuron]): Unit =
    for (neuron <- sequence.iterator if !skip.contains(neuron))
      neuron.delta =
        neuron.connectionsOut.foldLeft(0d) { (sum, connection) =>
          val targetNeuron = connection.neuronTarget
          if (reachable.contains(targetNeuron))
            sum + connection.weight * targetNeuron.delta / targetNeuron.connectionsIn.length
          else sum
        } * neuron.activationDerivative(neuron.preActivation) // outgoing sum * activation derivative

  // Turns the deltas into parameter updates - every delta must have been calculated before.
  // maxUpdateNorm caps the length of the whole update vector, leaving its direction
  // alone since it is the extreme steps that blow the net.
  // Indexed loops are intentional here because this pass visits every connection per example.
  def applyUpdates(order: GraphTraversal.ReverseOrder, learningRate: Double,
                   maxUpdateNorm: Double = Double.PositiveInfinity): Unit = {
    if (order.hasSharedParameters) {
      // Shared cells receive one summed, clipped update.
      val weights = order.connectionWeights
      val biases = order.neuronBiases
      var index = 0
      while (index < weights.length) {
        weights(index).accumulatedGradient = 0d
        index += 1
      }
      index = 0
      while (index < biases.length) {
        biases(index).accumulatedGradient = 0d
        index += 1
      }

      val sequence = order.sequence
      var neuronIndex = sequence.length - 1
      while (neuronIndex >= 0) {
        val neuron = sequence(neuronIndex)
        val connectionsIn = neuron.connectionsIn
        val fanIn = connectionsIn.length
        val neuronDelta = neuron.delta
        var connectionIndex = 0
        while (connectionIndex < fanIn) {
          val connection = connectionsIn(connectionIndex)
          val weight = connection.weightCell
          weight.accumulatedGradient += neuronDelta * connection.neuronSource.value / fanIn
          connectionIndex += 1
        }
        if (fanIn > 0)
          neuron.biasCell.accumulatedGradient += neuronDelta
        neuronIndex -= 1
      }

      var weightGradientSquares = 0d
      index = 0
      while (index < weights.length) {
        val gradient = weights(index).accumulatedGradient
        weightGradientSquares += gradient * gradient
        index += 1
      }
      var biasGradientSquares = 0d
      index = 0
      while (index < biases.length) {
        val gradient = biases(index).accumulatedGradient
        biasGradientSquares += gradient * gradient
        index += 1
      }
      val norm = math.sqrt(weightGradientSquares + biasGradientSquares)
      val scale = if (norm > maxUpdateNorm) maxUpdateNorm / norm else 1d
      val scaledLearningRate = learningRate * scale

      index = 0
      while (index < weights.length) {
        val weight = weights(index)
        weight.value += scaledLearningRate * weight.accumulatedGradient
        weight.accumulatedGradient = 0d
        index += 1
      }
      index = 0
      while (index < biases.length) {
        val bias = biases(index)
        bias.value += scaledLearningRate * bias.accumulatedGradient
        bias.accumulatedGradient = 0d
        index += 1
      }
    } else {
      val sequence = order.sequence
      val scale =
        if (maxUpdateNorm.isPosInfinity) 1d
        else {
          var sumSquares = 0d
          var neuronIndex = sequence.length - 1
          while (neuronIndex >= 0) {
            // caching values in local variables since this is a very performance critical section
            val neuron = sequence(neuronIndex)
            val connectionsIn = neuron.connectionsIn
            val fanIn = connectionsIn.length
            val neuronDelta = neuron.delta
            var connectionIndex = 0
            while (connectionIndex < fanIn) {
              val connection = connectionsIn(connectionIndex)
              val gradient = neuronDelta * connection.neuronSource.value / fanIn
              sumSquares += gradient * gradient
              connectionIndex += 1
            }
            if (fanIn > 0) sumSquares += neuronDelta * neuronDelta
            neuronIndex -= 1
          }
          val norm = math.sqrt(sumSquares)
          if (norm > maxUpdateNorm) maxUpdateNorm / norm else 1d
        }

      val scaledLearningRate = learningRate * scale
      var neuronIndex = sequence.length - 1
      while (neuronIndex >= 0) {
        val neuron = sequence(neuronIndex)
        val connectionsIn = neuron.connectionsIn
        val fanIn = connectionsIn.length
        val neuronDelta = neuron.delta
        var connectionIndex = 0
        while (connectionIndex < fanIn) {
          val connection = connectionsIn(connectionIndex)
          connection.weight += scaledLearningRate * neuronDelta * connection.neuronSource.value / fanIn
          connectionIndex += 1
        }
        if (fanIn > 0) // skip input neurons
          neuron.bias += scaledLearningRate * neuronDelta
        neuronIndex -= 1
      }
    }
  }
}
