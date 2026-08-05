package elaio.neuralnet.connections

import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache

trait Connection {
  protected var _neuronSource: Neuron
  protected var _neuronTarget: Neuron

  def neuronSource: Neuron = _neuronSource
  def neuronSource_=(neuron: Neuron): Unit = { _neuronSource = neuron }

  def neuronTarget: Neuron = _neuronTarget
  def neuronTarget_=(neuron: Neuron): Unit = { _neuronTarget = neuron }

  private var _id: Long = ConnectionCounter.getNext()

  def id: Long = _id

  // initialize weight based on a rendom number for now
  private var _weight: Double = scala.util.Random.nextDouble() * 2d - 1d

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
