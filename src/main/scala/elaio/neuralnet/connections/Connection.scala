package elaio.neuralnet.connections

import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache

// trait for a connection between two neurons
trait Connection(val id: Long) {
  protected var _neuronSource: Neuron
  protected var _neuronTarget: Neuron

  def neuronSource: Neuron = _neuronSource
  def neuronTarget: Neuron = _neuronTarget

  // initialize weight with 0 here since it is initialized later by WeightInitializer.
  // a cell of its own by default - sharing means handing the same cell to several connections
  private var _weight: Weight = new Weight

  def weight: Double = _weight.value
  def weight_=(value: Double): Unit = { _weight.value = value }

  def weightCell: Weight = _weight
  def weightCell_=(cell: Weight): Unit = { _weight = cell }

  def collect(cache: NeuronCollectionCache): Double = {
    // Cache accessors locally because this runs once per traversed connection.
    val source = neuronSource
    val cachedNeuron = cache.get(source.id)
    val neuronValue =
      if (cachedNeuron != null) {
        cachedNeuron.value
      } else {
        val value = source.collectInConnections(cache)
        cache.add(source)
        value
      }
    neuronValue * weight
  }
}
