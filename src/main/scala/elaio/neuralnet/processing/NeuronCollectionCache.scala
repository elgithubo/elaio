package elaio.neuralnet.processing

import scala.collection.mutable.HashMap
import elaio.neuralnet.units.Neuron

object NeuronCollectionCache {
  private var cache: HashMap[Double, Neuron] = HashMap.empty[Double, Neuron]
  private var stack = Array[HashMap[Double, Neuron]]()

  def clear() = {
    cache = HashMap.empty[Double, Neuron]
  }

  // def push() = {
  //   stack :+ cache
  // }
  
  //def pop() = {
    //cache = stack[]
    //stack -= cache
  //}
  
  def add(neuron: Neuron) = {
    cache ++= List(neuron.id -> neuron)
  }

  def get(id: Double): Neuron = {
    if (cache.contains(id)) {
      cache(id)
    } else { null }
  }
}
