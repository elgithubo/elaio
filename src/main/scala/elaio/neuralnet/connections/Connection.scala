package elaio.neuralnet.connections

import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache

trait Connection(val id: Long) {
  protected var _neuronSource: Neuron
  protected var _neuronTarget: Neuron

  def neuronSource: Neuron = _neuronSource
  def neuronSource_=(neuron: Neuron): Unit = { _neuronSource = neuron }

  def neuronTarget: Neuron = _neuronTarget
  def neuronTarget_=(neuron: Neuron): Unit = { _neuronTarget = neuron }

  // initialize weight with 0 here since it is initialized later by WeightInitializer
  private var _weight: Double = 0d

  def weight: Double = _weight
  def weight_=(value: Double): Unit = { _weight = value }

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
