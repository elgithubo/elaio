package elaio.neuralnet.units

import elaio.neuralnet.trace.NetTrace

class InputNeuron extends Neuron {

  protected var _initValue: Double = -1d
  
  override def collectInConnections(pullWeight: Double, backpropagation: Boolean): Double = {
    _value
  }    
  
  def initInput(initValue: Double, tolerance: Double): Unit = {
    _initValue = initValue
    _value = initValue
    init(tolerance)
  }

  def initValue: Double = {
    _initValue
  }
}

