package de.elaio.neuralnet.units

import de.elaio.neuralnet.connections.Connection
import de.elaio.neuralnet.trace.NetTrace

abstract class Neuron {

  val layer: BigInt = 0
  protected var _value: Float = 1.0f
  protected val _id: Int = NeuronCounter.getNext()

  //var connectionsOut: Array[Connection] = Array[Connection]()
  var connectionsIn: Array[Connection] = Array[Connection]()

  def value: Float = _value
  def id: Int = _id

  def value_(value: Float): Unit = {
    _value = value
  }

  def init(value: Float): Unit = {
    value_(value)
  }

  def collectInConnections(): Float = {

    var inValue: Float = 0
    if( connectionsIn.length == 0 ) {
      NetTrace.WriteMessage( "empty in for ID " + id )
    }
    for (connectionIn <- connectionsIn) {
      inValue = inValue + connectionIn.collect
    }
    _value = inValue //TODO consider old value
    _value
  }  

  def activationFunction(input: Float): Float = { // ReLu activation function as example
    if (input > 0) input else 0
  }

  // def addOutConnection(outConnection: Connection): Unit = {
  //   connectionsOut = connectionsOut :+ outConnection
  // }

  def addInConnection(inConnection: Connection): Unit = {
    connectionsIn = connectionsIn :+ inConnection
  }  

}
