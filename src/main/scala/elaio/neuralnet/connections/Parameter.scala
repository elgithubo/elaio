package elaio.neuralnet.connections

// One trainable scalar - a connection's weight or a neuron's bias. Several holders may point at the
// same cell, which is what turns a stack of identically built containers into one function applied
// to every token instead of N functions. Sharing needs no accumulation buffer: backpropagation adds
// in place and every delta is final before the update loop starts, so N in-place additions are the
// summed gradient.
final class Parameter(var value: Double = 0d)
