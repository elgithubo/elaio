package elaio.neuralnet.processing

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.units.{InputNeuron, Neuron, OutputNeuron}

// One input feeding two outputs that share a weight cell and a bias cell. The point of the
// clipping path is that it sizes an update by the unique parameters, not by the connections
// that happen to read them - two sharers must not count twice towards the norm.
class SharedParameterSuite extends munit.FunSuite {

  test("a clipped update over shared cells has exactly the capped norm") {
    val input = new InputNeuron(1L)
    val outputOne = new OutputNeuron(2L)
    val outputTwo = new OutputNeuron(3L)

    val connectionOne = new Connection(1L) {
      protected var _neuronSource: Neuron = input
      protected var _neuronTarget: Neuron = outputOne
    }
    val connectionTwo = new Connection(2L) {
      protected var _neuronSource: Neuron = input
      protected var _neuronTarget: Neuron = outputTwo
    }

    input.addOutConnection(connectionOne)
    input.addOutConnection(connectionTwo)
    outputOne.addInConnection(connectionOne)
    outputTwo.addInConnection(connectionTwo)
    connectionTwo.weightCell = connectionOne.weightCell
    outputTwo.biasCell = outputOne.biasCell

    input.initInput(2d)
    outputOne.initOutput(3d)
    outputTwo.initOutput(4d)
    outputOne.collectInConnections(new NeuronCollectionCache)
    outputTwo.collectInConnections(new NeuronCollectionCache)

    val order = GraphTraversal.reverseTopologicalFromOutputs(Array(outputOne, outputTwo))
    assertEquals(order.connectionWeights.length, 1)
    assertEquals(order.neuronBiases.length, 1)
    assert(order.hasSharedParameters)

    // the probe this came from called a Backpropagation.run that the delta and update phases replaced
    Backpropagation.seedDeltas(order)
    Backpropagation.calculateDeltas(order.sequence, order.outputs)
    Backpropagation.applyUpdates(order, learningRate = 1d, maxUpdateNorm = 1d)

    val updateNorm =
      math.sqrt(connectionOne.weight * connectionOne.weight + outputOne.bias * outputOne.bias)
    assertEqualsDouble(updateNorm, 1d, 1e-12)
  }
}
