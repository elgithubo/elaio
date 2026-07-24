package elaio.neuralnet.units

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.activation.Activation
import scala.compiletime.ops.boolean

abstract class Neuron {

  protected var _weight: Double = 0.5d
  protected var _value: Double = 1d
  protected var _tolerance: Double = 0.5d
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

  def init(tolerance: Double): Unit = {
    _tolerance = tolerance

    for(connectionOut <- connectionsOut) {
      connectionOut.getNeuronTarget._value = _weight * _value
    }
  }

  def collectInConnections(pullWeight: Double, backpropagation: Boolean): Double = {

    _weight = pullWeight

    var valueSum = 0d
    for (connectionIn <- connectionsIn) {
      valueSum = valueSum + connectionIn.collect(pullWeight, backpropagation)
    }
    if (connectionsIn.nonEmpty)
      valueSum = valueSum / connectionsIn.length

    _preActivation = valueSum

    if(backpropagation)
      _value = activationDerivative(valueSum)
    else
      _value = activationFunction(valueSum)

    _weight = 1.7976931348623157E308 - (_value * valueSum)

    //NetTrace.WriteMessage("collected Value: " + _value)
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
