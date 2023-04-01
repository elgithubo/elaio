package de.elaio.neuralnet.connections

import scala.math._
import de.elaio.neuralnet.units.Neuron
import de.elaio.neuralnet.processing.NeuronCollectionCache

trait Connection {

  val weight: Float
  val neuronSource: Neuron
  val neuronTarget: Neuron

  def collect: Float = {
    val cachedNeuron = NeuronCollectionCache.get(neuronSource.id)
    if (cachedNeuron != null) {
      val root = sqrt(cachedNeuron.value.floatValue)
      root.toFloat
    } else {
      val neuronValue = neuronSource.collectInConnections() * weight
      NeuronCollectionCache.add(neuronSource)
      neuronValue
    }
  }

  def getNeuronSource: Neuron = neuronSource

  def getNeuronTarget: Neuron = neuronTarget
}
