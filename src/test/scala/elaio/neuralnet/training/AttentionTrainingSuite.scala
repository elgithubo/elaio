package elaio.neuralnet.training

import elaio.neuralnet.test.AttentionTokenTest

// The shipped attention task shrunk to one epoch. It is the only test that drives the whole
// DepthAttention gradient path end to end - refined pass, first-pass recompute, seeded deltas,
// accumulated updates - so it is here to catch that path throwing or going non-finite.
class AttentionTrainingSuite extends munit.FunSuite {

  private final class SmallAttentionTraining extends AttentionTokenTest(None) {
    override protected val epochs = 1
    // both have to stay divisible by the task's key count
    override protected val trainCount = 15
    override protected val numberOfQuestions = 15
  }

  test("one epoch of attention training runs through and stays finite") {
    val output = new java.io.ByteArrayOutputStream
    Console.withOut(output)(new SmallAttentionTraining().run())
    val trace = output.toString

    assert(trace.contains("end of test run"), "the run did not reach the end")

    // The probe this came from waited for a named worker pool to drain. Those passes run on the
    // global execution context now, whose threads are daemons, so finiteness is what is left.
    // Only the reported numbers may be searched for it: the run's own banner carries the word NaN,
    // so matching the whole trace would fail on every run no matter how healthy it was.
    val reportedNumbers =
      trace.linesIterator.filter(line => line.contains("total squared error =") || line.contains("   target ")).toVector
    // a filter that matches nothing would make the check below pass without testing anything
    assert(reportedNumbers.nonEmpty, "the run reported neither an epoch error nor an output value")

    val nonFinite = reportedNumbers.filter(line => line.contains("NaN") || line.contains("Infinity"))
    assert(nonFinite.isEmpty, "attention training went non-finite:\n" + nonFinite.mkString("\n"))
  }
}
