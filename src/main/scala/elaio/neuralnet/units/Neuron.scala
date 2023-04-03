package elaio.neuralnet.units

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.trace.NetTrace

abstract class Neuron {

  //val layer: BigInt = 0
  protected var _value: Double = id.toDouble
  protected var _target: Double = 1d
  protected var _tolerance: Double = 0.5d
  protected val _id: Double = NeuronCounter.getNext()

  var connectionsOut: Array[Connection] = Array[Connection]()
  var connectionsIn: Array[Connection] = Array[Connection]()

  def value: Double = _value
  def id: Double = _id

  def value_(value: Double): Unit = {
    _value = value
  }

  def init(value: Double, target: Double, tolerance: Double): Unit = {
    value_(value)
    //NetTrace.WriteMessage( "init: " + target + " - " + tolerance )
    _target = target
    _tolerance = tolerance
  }

  def collectInConnections(): Double = {
    var inValue: Double = 0
      
    if(_value > _target - _tolerance && _value < _target + _tolerance ) {
      NetTrace.WriteMessage( "found value: " + _value + " min: " + (_target - _tolerance) + "max: " + ( _target + _tolerance ) )
      return _value
    }

    for (connectionIn <- connectionsIn) {
      var subnodeValue = connectionIn.collect
      if(subnodeValue != 1.0d && subnodeValue > _target - _tolerance && subnodeValue < _target + _tolerance ) { //TODO remove hardcoded one
        //NetTrace.WriteMessage( "found subnode: " + subnodeValue + " min: " + (_target - _tolerance) + "max: " + ( _target + _tolerance ) )
        _value = subnodeValue
        return _value
      }
      inValue = inValue + subnodeValue
    }
    if(inValue > _target - _tolerance && inValue < _target + _tolerance ) {
      //NetTrace.WriteMessage( "found collected: " + inValue + " min: " + (_target - _tolerance) + "max: " + ( _target + _tolerance ) )
    }
    
    //NetTrace.WriteMessage( "collected: " + inValue + " min: " + (_target - _tolerance) + "max: " + ( _target + _tolerance ) )
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
