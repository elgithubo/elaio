package elaio.neuralnet.connections
import scala.math.sqrt
import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache
trait Connection{
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
