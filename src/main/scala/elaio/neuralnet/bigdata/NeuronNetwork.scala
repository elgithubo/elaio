package elaio.neuralnet.bigdata

import elaio.neuralnet.TokenMatrix
import elaio.neuralnet.processing.GraphTraversal
import elaio.neuralnet.units.{InputNeuron, OutputNeuron}

// What training needs from a built network, regardless of what built it.
trait NeuronNetwork {
  def inputNodes: Array[InputNeuron]
  def outputNodes: Array[OutputNeuron]
  def reverseOrder: GraphTraversal.ReverseOrder

  // Feed one example. Only the network knows whether its input is one row or a matrix, so the
  // token rows are resolved here and nowhere else - a single container joins them, a stack hands
  // one row to each of its containers.
  def initInputs(tokens: TokenMatrix): Unit
}
