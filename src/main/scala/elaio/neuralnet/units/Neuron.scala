package elaio.neuralnet.units

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.trace.NetTrace

abstract class Neuron {

  //val layer: BigInt = 0
  protected var _value: Double = id.toDouble
  protected val _id: Double = NeuronCounter.getNext()

  var connectionsOut: Array[Connection] = Array[Connection]()
  var connectionsIn: Array[Connection] = Array[Connection]()

  def value: Double = _value
  def id: Double = _id

  def value_(value: Double): Unit = {
    _value = value
  }

  def init(value: Double): Unit = {
    value_(value)
  }

  def collectInConnections(): Double = {

    var inValue: Double = 0

    for (connectionIn <- connectionsIn) {
      inValue = inValue + connectionIn.collect
    }
    //NetTrace.WriteMessage( "collected: " + inValue )
    _value = inValue
    _value
  }  

  def activationFunction(input: Double): Double = {
    if (input > 0) input else 1
  }

  def addOutConnection(outConnection: Connection): Unit = {
    connectionsOut = connectionsOut :+ outConnection
  }

  def addInConnection(inConnection: Connection): Unit = {
    connectionsIn = connectionsIn :+ inConnection
  }  

}
