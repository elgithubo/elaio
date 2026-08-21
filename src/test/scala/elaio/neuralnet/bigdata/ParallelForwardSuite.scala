package elaio.neuralnet.bigdata

import elaio.neuralnet.processing.NeuronCollectionCache
import elaio.neuralnet.training.WeightInitializer
import elaio.neuralnet.units.NeuronDataCreator

// The pooled forward hands one container per token to a Future, so it rests entirely on the
// happens-before edges of submit-then-await documented on NeuronNetwork.forward. It has to land
// on the number the single-cache recursion produces, and land there every time.
class ParallelForwardSuite extends munit.FunSuite {

  test("pooled forward matches the serial collect bit-exactly and repeats") {
    val layered = new LayeredContainer(4, 2, 5, 5, 3, new NeuronDataCreator)
    layered.init()
    WeightInitializer.initialize(layered.reverseOrder)

    val random = new scala.util.Random(42)
    // one row per container - this stack was built for two tokens
    val tokens = Array.fill(2)(Array.fill(5)(random.nextDouble() * 10 - 5))
    layered.initInputs(tokens)

    // reference: the serial path - one cache, full recursive collect from the read-out
    val serialCache = new NeuronCollectionCache
    for (outputNode <- layered.outputNodes) outputNode.collectInConnections(serialCache)
    val serial = layered.outputNodes.map(_.value).toSeq

    val cache = new NeuronCollectionCache
    layered.forward(cache)
    val parallelFirst = layered.outputNodes.map(_.value).toSeq
    layered.forward(cache)
    val parallelSecond = layered.outputNodes.map(_.value).toSeq

    assertEquals(parallelFirst, serial, "pooled forward diverges from the serial forward")
    assertEquals(parallelSecond, parallelFirst, "pooled forward is not deterministic")
  }
}
