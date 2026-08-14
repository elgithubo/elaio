package elaio.neuralnet.processing

import scala.collection.mutable.LongMap
import elaio.neuralnet.units.Neuron

// Reusable within one candidate and thread-confined
class NeuronCollectionCache {
  // performance tweak: LongMap avoids boxing IDs on the hot path.
  private val cache: LongMap[Neuron] = LongMap.empty[Neuron]

  def clear(): Unit = cache.clear()

  def add(neuron: Neuron): Unit = cache(neuron.id) = neuron

  def get(id: Long): Neuron = cache.getOrElse(id, null)

}
