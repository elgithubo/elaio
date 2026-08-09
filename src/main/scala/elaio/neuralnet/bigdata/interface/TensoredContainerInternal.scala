package elaio.neuralnet.bigdata.interface

import elaio.neuralnet.units.{HiddenNeuron, Neuron}

class TensoredContainerInternal {
  protected var _inputNodes: Array[Neuron] = Array.empty[Neuron]
  protected var _outputNodes: Array[Neuron] = Array.empty[Neuron]
  protected var _intermediateNodes: Array[HiddenNeuron] = Array.empty[HiddenNeuron]

  def inputNodes: Array[Neuron] = _inputNodes
  def outputNodes: Array[Neuron] = _outputNodes
  def intermediateNodes: Array[HiddenNeuron] = _intermediateNodes

  def inputNodes_=(nodes: Array[Neuron]): Unit =
    _inputNodes = nodes

  def addInputNode(node: Neuron): Unit =
    _inputNodes :+= node

  def addOutputNode(node: Neuron): Unit =
    _outputNodes :+= node

  def intermediateNodes_=(nodes: Array[HiddenNeuron]): Unit =
    _intermediateNodes = nodes

  // appends - not a setter, so no _= here
  def addIntermediateNodes(nodes: Array[HiddenNeuron]): Unit =
    _intermediateNodes ++= nodes
}
