package elaio.neuralnet.processing

import scala.collection.mutable.HashMap
import elaio.neuralnet.units.Neuron

// one instance per forward pass, confined to the calling thread
class NeuronCollectionCache {
  private val cache: HashMap[Long, Neuron] = HashMap.empty[Long, Neuron]

  def add(neuron: Neuron): Unit = cache(neuron.id) = neuron

  def get(id: Long): Neuron = cache.getOrElse(id, null)

}
