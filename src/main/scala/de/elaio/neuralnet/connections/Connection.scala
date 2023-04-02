package de.elaio.neuralnet.connections

import scala.math._
import de.elaio.neuralnet.units.Neuron
import de.elaio.neuralnet.processing.NeuronCollectionCache

trait Connection {

  val weight: Double
  val neuronSource: Neuron
  val neuronTarget: Neuron

  def collect: Double = {
    val cachedNeuron = NeuronCollectionCache.get(neuronSource.id)
    if (cachedNeuron != null) {
      sqrt(cachedNeuron.value)
    } else {
      val neuronValue = neuronSource.collectInConnections() * weight
      NeuronCollectionCache.add(neuronSource)
      sqrt(neuronValue)
    }
  }

  def getNeuronSource: Neuron = neuronSource

  def getNeuronTarget: Neuron = neuronTarget
}
