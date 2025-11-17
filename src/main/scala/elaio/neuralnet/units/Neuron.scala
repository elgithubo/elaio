package elaio.neuralnet.units

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.activation.Activation
import scala.compiletime.ops.boolean

abstract class Neuron {

  protected var _weight: Double = 0.5d
  protected var _value: Double = 1d
  protected var _tolerance: Double = 0.5d
  protected val _id: Double = NeuronCounter.getNext()

  var connectionsOut: Array[Connection] = Array[Connection]()
  var connectionsIn: Array[Connection] = Array[Connection]()

  def value: Double = _value
  def id: Double = _id

  def init(tolerance: Double): Unit = {
    _tolerance = tolerance

    for(connectionOut <- connectionsOut) {
      connectionOut.getNeuronTarget._value = _weight * _value
    }
  }


/*
  def tolerance: Double =  {
    _tolerance
  }  
*/

  def collectInConnections(pullWeight: Double, backpropagation: Boolean): Double = {

    _weight = pullWeight

    var valueSum = 0d
    for (connectionIn <- connectionsIn) {
      valueSum = connectionIn.collect(pullWeight, backpropagation)
    }

    if(backpropagation)
      _value = Activation.backpropagationFunction(_value)
    else
      _value = Activation.activationFunction(_value)

    _weight = 1.7976931348623157E308 - (_value * valueSum)
    _value
  }  

  def activationFunction(input: Double): Double = {
    Activation.activationFunction(input)
  }

  def addOutConnection(outConnection: Connection): Unit = {
    connectionsOut = connectionsOut :+ outConnection
  }

  def addInConnection(inConnection: Connection): Unit = {
    connectionsIn = connectionsIn :+ inConnection
  }  

}
