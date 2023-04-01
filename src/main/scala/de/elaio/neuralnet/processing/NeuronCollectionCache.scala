package de.elaio.neuralnet.processing

import scala.collection.mutable.HashMap
import de.elaio.neuralnet.units.Neuron

object NeuronCollectionCache {
  private var cache: HashMap[Int, Neuron] = HashMap.empty[Int, Neuron]

  def clear() = {
    cache = HashMap.empty[Int, Neuron]
  }

  def add(neuron: Neuron) = {
    cache ++= List(neuron.id -> neuron)
  }

  def get(id: Int): Neuron = {
    if (cache.contains(id)) {
      cache(id)
    } else { null }
  }
}
