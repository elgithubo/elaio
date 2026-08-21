package elaio.neuralnet.bigdata

import elaio.neuralnet.units.{IntermediateOutputNeuron, NeuronDataCreator}

// externalOutWidth is what a network exposes to its caller. tokenOutWidth only takes effect once
// tokens are pooled, which is why a stack of one ignores it - the 99 below has to stay unused.
class OutputWidthSuite extends munit.FunSuite {

  private def connectionCount(container: LayeredContainer): Int =
    container.reverseOrder.sequence.iterator.map(_.connectionsIn.length).sum

  test("a pooled stack exposes externalOutWidth and keeps a token rank behind it") {
    val attention = new LayeredContainer(2, 4, 5, 5, 1, new NeuronDataCreator)
    attention.init()

    assertEquals(attention.inputNodes.length, 20)
    assertEquals(attention.outputNodes.length, 1)
    assertEquals(attention.reverseOrder.sequence.count(_.isInstanceOf[IntermediateOutputNeuron]), 20)
  }

  test("a stack of one exposes its own outputs and builds no intermediate rank") {
    val calculator = new LayeredContainer(2, 1, 5, 99, 4, new NeuronDataCreator)
    calculator.init()

    assertEquals(calculator.inputNodes.length, 5)
    assertEquals(calculator.outputNodes.length, 4)
    assert(calculator.reverseOrder.sequence.forall(!_.isInstanceOf[IntermediateOutputNeuron]))
  }

  test("a pass-through stack of one keeps the bare container's size") {
    val division = new LayeredContainer(2, 1, 5, 5, 5, new NeuronDataCreator)
    division.init()

    assertEquals(division.reverseOrder.sequence.length, 78)
    assertEquals(connectionCount(division), 1231)
  }
}
