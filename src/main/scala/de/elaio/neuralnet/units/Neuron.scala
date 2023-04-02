package de.elaio.neuralnet.units

import de.elaio.neuralnet.connections.Connection
import de.elaio.neuralnet.trace.NetTrace

abstract class Neuron {

  val layer: BigInt = 0
  protected var _value: Double = id.toDouble
  protected val _id: Int = NeuronCounter.getNext()

  //var connectionsOut: Array[Connection] = Array[Connection]()
  var connectionsIn: Array[Connection] = Array[Connection]()

  def value: Double = _value
  def id: Int = _id

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
    NetTrace.WriteMessage( "collected: " + inValue )
    _value = inValue
    _value
  }  

  def activationFunction(input: Double): Double = { // ReLu activation function as example
    if (input > 0) input else 1
  }

  //def addOutConnection(outConnection: Connection): Unit = {
  //  connectionsOut = connectionsOut :+ outConnection
  //}

  def addInConnection(inConnection: Connection): Unit = {
    connectionsIn = connectionsIn :+ inConnection
  }  

}
