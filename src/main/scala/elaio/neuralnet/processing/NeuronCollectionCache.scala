package elaio.neuralnet.processing

import scala.collection.mutable.HashMap
import elaio.neuralnet.units.Neuron

object NeuronCollectionCache {
  // reused across forward passes: clear() empties it instead of allocating a new
  // map, because this runs once per example per epoch
  private val cache: HashMap[Double, Neuron] = HashMap.empty[Double, Neuron]

  def clear(): Unit = cache.clear()

  def add(neuron: Neuron): Unit = cache(neuron.id) = neuron

  def get(id: Double): Neuron = cache.getOrElse(id, null)

  def size: Int = cache.size
}
