package elaio.neuralnet.processing

import scala.collection.mutable.{ArrayBuffer, HashMap}
import elaio.neuralnet.units.Neuron

object NeuronCollectionCache {
  private var cache: HashMap[Double, Neuron] = HashMap.empty[Double, Neuron]
  // the stack records the order neurons finish computing in during a forward pass.
  // since a neuron is only added once all of its own sources have already been added,
  // this order is a valid topological order - reversing it gives a valid order
  // for a backward (backpropagation) pass with no separate graph walk needed.
  private var stack = Array[HashMap[Double, Neuron]]()
  private var order: ArrayBuffer[Neuron] = ArrayBuffer.empty[Neuron]

  def clear() = {
    cache = HashMap.empty[Double, Neuron]
    order = ArrayBuffer.empty[Neuron]
  }

  def add(neuron: Neuron) = {
    cache ++= List(neuron.id -> neuron)
    order += neuron
  }

  def get(id: Double): Neuron = {
    if (cache.contains(id)) {
      cache(id)
    } else { null }
  }

  def size: Int = cache.size

  def visitedInOrder: ArrayBuffer[Neuron] = order
}
