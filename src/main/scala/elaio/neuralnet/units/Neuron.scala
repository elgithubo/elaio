package elaio.neuralnet.units

import scala.collection.mutable
import elaio.neuralnet.connections.Connection
import elaio.neuralnet.processing.NeuronCollectionCache

abstract class Neuron(val id: Long) {

  protected var _value: Double = 1d
  protected var _preActivation: Double = 0d
  protected var _delta: Double = 0d
  // starts at 0, so a fresh net behaves exactly as it did before biases existed.
  // Without it the net is positively homogeneous - N(c*x) = c*N(x), measured as
  // exactly 2.000 - and can only ever represent maps that scale linearly.
  protected var _bias: Double = 0d
  private var _attentionContext: Double = 0d


  protected val _connectionsOut: mutable.ArrayBuffer[Connection] = mutable.ArrayBuffer.empty
  protected val _connectionsIn: mutable.ArrayBuffer[Connection] = mutable.ArrayBuffer.empty

  def value: Double = _value
  def preActivation: Double = _preActivation
  def delta: Double = _delta
  def delta_=(delta: Double): Unit = { _delta = delta }
  def bias: Double = _bias
  def bias_=(bias: Double): Unit = { _bias = bias }
  private[neuralnet] def attentionContext_=(value: Double): Unit = { _attentionContext = value }
  // addInConnection and addOutConnection are the only way to wire a neuron
  def connectionsOut: scala.collection.IndexedSeq[Connection] = _connectionsOut
  def connectionsIn: scala.collection.IndexedSeq[Connection] = _connectionsIn


  def activationFunction(input: Double): Double
  def activationDerivative(input: Double): Double

  // one forward pass: pull from every in-connection, average, offset, activate.
  def collectInConnections(cache: NeuronCollectionCache): Double = {

    var valueSum = 0d
    for (connectionIn <- connectionsIn)
      valueSum = valueSum + connectionIn.collect(cache)

    if (connectionsIn.nonEmpty)
      valueSum = valueSum / connectionsIn.length

    // the bias is added after the averaging - it is an independent offset, not
    // one more incoming value to average in
    _preActivation = valueSum + _bias + _attentionContext
    _value = activationFunction(_preActivation)

    _value
  }

  def addOutConnection(outConnection: Connection): Unit = {
    _connectionsOut += outConnection
  }

  def addInConnection(inConnection: Connection): Unit = {
    _connectionsIn += inConnection
  }

}
