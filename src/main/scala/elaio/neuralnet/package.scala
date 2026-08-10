package elaio.neuralnet

// one example: rows are tokens, columns are the values of a token.
// lives here rather than in a single package because the networks read it and the tests write it.
type TokenMatrix = Array[Array[Double]]
