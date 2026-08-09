package elaio.neuralnet.bigdata.interface

import elaio.neuralnet.units.{InputNeuron, OutputNeuron}

class TensoredContainerInOut {
  protected var _inputNodes: Array[InputNeuron] = Array.empty[InputNeuron]
  protected var _outputNodes: Array[OutputNeuron] = Array.empty[OutputNeuron]

  def inputNodes: Array[InputNeuron] = _inputNodes
  def outputNodes: Array[OutputNeuron] = _outputNodes

  def inputNodes_(nodes: Array[InputNeuron]): Unit =
    _inputNodes = nodes

  def outputNodes_(nodes: Array[OutputNeuron]): Unit =
    _outputNodes = nodes

  def addInputNode(node: InputNeuron): Unit =
    _inputNodes :+= node

  def addOutputNode(node: OutputNeuron): Unit =
    _outputNodes :+= node
}
