package elaio.neuralnet.units

import elaio.neuralnet.connections.Connection
import elaio.neuralnet.trace.NetTrace

abstract class Neuron {

  protected var _weight: Double = 1d
  protected var _value: Double = 0d
  protected var _target: Double = 6d
  protected var _tolerance: Double = 0.5d
  protected var _initValue: Double = -1
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
    _target = target
    _tolerance = tolerance
  }

  def initValue: Double =  {
    _initValue
  }


  def target: Double =  {
    _target
  }

  def collectInConnections(PullWeight: Double): Double = {
    var inValue: Double = 0
    var checkValue: Double = 0

    _weight = PullWeight
      
    if(_value > _target - _tolerance && _value < _target + _tolerance) {
      return _value
    }
    var doExit: Boolean = false
    for (connectionIn <- connectionsIn) {
      if(doExit == false) {
        var subnodeValue = connectionIn.collect(PullWeight)
        NetTrace.WriteMessage( "collected subnode: " + subnodeValue + " min: " + (_target - _tolerance) + " - max: " + ( _target + _tolerance ) )
        if(doExit == false) {
          checkValue = subnodeValue * subnodeValue
          if(checkValue != _target && checkValue > _target - _tolerance && checkValue < _target + _tolerance ) {
            NetTrace.WriteMessage( "found subnode: " + checkValue + " min: " + (_target - _tolerance) + " - max: " + ( _target + _tolerance ) )
            _value = checkValue
            doExit = true
          } 
          inValue = inValue + subnodeValue                   
        }
      }
    }
    if(doExit == false) {
      _value = inValue
    }
    _value
  }  

  def activationFunction(input: Double): Double = {
    if (input > 0) input else 0
  }

  def addOutConnection(outConnection: Connection): Unit = {
    connectionsOut = connectionsOut :+ outConnection
  }

  def addInConnection(inConnection: Connection): Unit = {
    connectionsIn = connectionsIn :+ inConnection
  }  

}
