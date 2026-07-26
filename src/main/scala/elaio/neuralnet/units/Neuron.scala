package elaio.neuralnet.units

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.activation.Activation

abstract class Neuron {

  protected var _value: Double = 1d
  protected var _preActivation: Double = 0d
  protected var _delta: Double = 0d
  protected val _id: Double = NeuronCounter.getNext()

  var connectionsOut: Array[Connection] = Array[Connection]()
  var connectionsIn: Array[Connection] = Array[Connection]()

  def value: Double = _value
  def id: Double = _id
  def preActivation: Double = _preActivation
  def delta: Double = _delta
  def delta_(delta: Double): Unit = { _delta = delta }

  // One forward pass: pull from every in-connection, average, activate.
  def collectInConnections(): Double = {

    var valueSum = 0d
    for (connectionIn <- connectionsIn) {
      valueSum = valueSum + connectionIn.collect()
    }
    if (connectionsIn.nonEmpty)
      valueSum = valueSum / connectionsIn.length

    _preActivation = valueSum
    _value = activationFunction(valueSum)

    _value
  }

  def activationFunction(input: Double): Double = {
    Activation.activationFunction(input)
  }

  def activationDerivative(input: Double): Double = {
    Activation.backpropagationFunction(input)
  }

  def addOutConnection(outConnection: Connection): Unit = {
    connectionsOut = connectionsOut :+ outConnection
  }

  def addInConnection(inConnection: Connection): Unit = {
    connectionsIn = connectionsIn :+ inConnection
  }  

}
