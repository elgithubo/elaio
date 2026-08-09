package elaio.neuralnet.bigdata

import elaio.neuralnet.processing.GraphTraversal
import elaio.neuralnet.units.{InputNeuron, OutputNeuron}

// What training needs from a built network, regardless of what built it.
trait NeuronNetwork {
  def inputNodes: Array[InputNeuron]
  def outputNodes: Array[OutputNeuron]
  def reverseOrder: GraphTraversal.ReverseOrder
}
