package de.elaio.neuralnet.processing

import scala.collection.mutable.HashMap
import de.elaio.neuralnet.units.Neuron

object NeuronCollectionCache {
  private var cache: HashMap[Double, Neuron] = HashMap.empty[Double, Neuron]

  def clear() = {
    cache = HashMap.empty[Double, Neuron]
  }

  def add(neuron: Neuron) = {
    cache ++= List(neuron.id -> neuron)
  }

  def get(id: Double): Neuron = {
    if (cache.contains(id)) {
      cache(id)
    } else { null }
  }
}
