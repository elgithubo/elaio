package elaio.neuralnet.connections

import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache

// trait for a connection between two neurons
trait Connection(val id: Long) {
  protected var _neuronSource: Neuron
  protected var _neuronTarget: Neuron

  def neuronSource: Neuron = _neuronSource
  def neuronSource_=(neuron: Neuron): Unit = { _neuronSource = neuron }

  def neuronTarget: Neuron = _neuronTarget
  def neuronTarget_=(neuron: Neuron): Unit = { _neuronTarget = neuron }

  // initialize weight with 0 here since it is initialized later by WeightInitializer.
  // a cell of its own by default - sharing means handing the same cell to several connections
  private var _weight: Parameter = new Parameter

  def weight: Double = _weight.value
  def weight_=(value: Double): Unit = { _weight.value = value }

  def weightParameter: Parameter = _weight
  def weightParameter_=(cell: Parameter): Unit = { _weight = cell }

  def collect(cache: NeuronCollectionCache): Double = {
    val cachedNeuron = cache.get(neuronSource.id)
    val neuronValue =
      if (cachedNeuron != null) {
        cachedNeuron.value
      } else {
        val v = neuronSource.collectInConnections(cache)
        cache.add(neuronSource)
        v
      }
    neuronValue * weight
  }
}
