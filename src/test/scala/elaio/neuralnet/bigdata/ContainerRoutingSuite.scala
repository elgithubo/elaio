package elaio.neuralnet.bigdata

import elaio.neuralnet.persistence.NetworkStateMapper
import elaio.neuralnet.training.WeightInitializer
import elaio.neuralnet.units.{IntermediateOutputNeuron, NeuronDataCreator, OutputNeuron}

// Where a LayeredContainer routes its outputs: straight through for a stack of one, through an
// intermediate rank and a read-out once tokens are pooled.
class ContainerRoutingSuite extends munit.FunSuite {

  private def connectionCount(container: NeuronNetwork): Int =
    container.reverseOrder.sequence.iterator.map(_.connectionsIn.length).sum

  // A stack of one is built without a read-out rank, so it has to be the very graph a bare
  // TensoredContainer builds - that equality is what lets every task run through LayeredContainer.
  test("a stack of one is the same graph as a bare container") {
    val plain = new TensoredContainer(2, 5, 5, new NeuronDataCreator)
    plain.init()
    val wrapped = new LayeredContainer(2, 1, 5, 99, 5, new NeuronDataCreator)
    wrapped.init()

    assertEquals(wrapped.reverseOrder.sequence.map(_.id), plain.reverseOrder.sequence.map(_.id))
    assertEquals(connectionCount(wrapped), connectionCount(plain))
    assertEquals(wrapped.reverseOrder.sequence.length, 78)
    assertEquals(connectionCount(wrapped), 1231)
  }

  // Pooling turns each container's own outputs into intermediates and adds the one read-out rank.
  test("a pooled stack has one intermediate output per token channel plus the read-out") {
    val pooled = new LayeredContainer(4, 2, 5, 5, 1, new NeuronDataCreator)
    pooled.init()
    val neurons = pooled.reverseOrder.sequence

    assertEquals(neurons.count(_.isInstanceOf[IntermediateOutputNeuron]), 10)
    // IntermediateOutputNeuron extends OutputNeuron, so the read-out is the one extra
    assertEquals(neurons.count(_.isInstanceOf[OutputNeuron]), 11)
  }

  test("capture and restore round-trip every shared weight and bias of a pooled stack") {
    val pooled = new LayeredContainer(4, 2, 5, 5, 1, new NeuronDataCreator)
    pooled.init()
    WeightInitializer.initialize(pooled.reverseOrder)

    val restored = new LayeredContainer(4, 2, 5, 5, 1, new NeuronDataCreator)
    restored.init()
    NetworkStateMapper.restore(NetworkStateMapper.capture(pooled), restored)

    assertEquals(
      restored.reverseOrder.connectionWeights.map(_.value),
      pooled.reverseOrder.connectionWeights.map(_.value)
    )
    assertEquals(
      restored.reverseOrder.neuronBiases.map(_.value),
      pooled.reverseOrder.neuronBiases.map(_.value)
    )
  }
}
