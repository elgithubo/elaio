package elaio.neuralnet.units

import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.activation.Activation
import elaio.neuralnet.processing.NeuronCollectionCache

class OutputNeuron() extends Neuron {

  protected var _target: Double = 0d

  //NeuronCollectionCache.clear()
  /*if(_value > 0)
    collectInConnections(1d, false)
  else
    collectInConnections(0d, false)*/

  def initOutput(target: Double, tolerance: Double): Unit = {
    _target = target
    NetTrace.WriteMessage(
      "initializing input neuron - target: " + target
    )
    init(tolerance)
  }

  override def collectInConnections(pullWeight: Double, backpropagation: Boolean): Double = {
    //NeuronCollectionCache.clear()
    _target - _value
  }

  def target: Double =  {
    _target
  }
}
