package elaio.neuralnet.connections

// One trainable weight. Several connections may point at the same cell - that is what turns a stack
// of identically built containers into one function applied to every token, instead of N functions.
// Sharing needs no accumulation buffer: backpropagation adds to the weight in place and every delta
// is already final when the update loop starts, so N in-place additions are the summed gradient.
final class Weight(var value: Double = 0d) {
  private[neuralnet] var accumulatedGradient: Double = 0d
}
