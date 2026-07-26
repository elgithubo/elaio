package elaio.neuralnet.processing

import scala.collection.mutable.{ArrayBuffer, HashMap}
import elaio.neuralnet.units.Neuron

object NeuronCollectionCache {
  private var cache: HashMap[Double, Neuron] = HashMap.empty[Double, Neuron]
  // A neuron is only added once all its sources are, so order is a topological
  // order - reversed, it is a valid backpropagation order.
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
