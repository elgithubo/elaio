package elaio.neuralnet.units

import scala.collection.mutable
import elaio.neuralnet.connections.Connection
import elaio.neuralnet.activation.Activation

abstract class Neuron {

  protected var _value: Double = 1d
  protected var _preActivation: Double = 0d
  protected var _delta: Double = 0d
  protected val _id: Double = NeuronCounter.getNext()

  // a buffer, not an Array: appending to an Array with :+ copies the whole thing
  // every time, which makes building a neuron quadratic in its fan-in
  protected val _connectionsOut: mutable.ArrayBuffer[Connection] = mutable.ArrayBuffer.empty
  protected val _connectionsIn: mutable.ArrayBuffer[Connection] = mutable.ArrayBuffer.empty

  def value: Double = _value
  def id: Double = _id
  def preActivation: Double = _preActivation
  def delta: Double = _delta
  def delta_=(delta: Double): Unit = { _delta = delta }
  // addInConnection and addOutConnection are the only way to wire a neuron
  def connectionsOut: scala.collection.IndexedSeq[Connection] = _connectionsOut
  def connectionsIn: scala.collection.IndexedSeq[Connection] = _connectionsIn

  // one forward pass: pull from every in-connection, average, activate.
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
    _connectionsOut += outConnection
  }

  def addInConnection(inConnection: Connection): Unit = {
    _connectionsIn += inConnection
  }

}
