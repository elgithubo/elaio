package elaio.neuralnet.connections
import scala.math.sqrt
import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache
trait Connection{
  val neuronSource: Neuron
  val neuronTarget: Neuron
  def collect(pullWeight: Double, backpropagation: Boolean): Double = {
    val cachedNeuron = NeuronCollectionCache.get(neuronSource.id)
    if (cachedNeuron != null) {
      cachedNeuron.value
    } else {
      val neuronValue = neuronSource.collectInConnections(pullWeight, backpropagation)
      NeuronCollectionCache.add(neuronSource)
      neuronValue
    }
  }
  def getNeuronSource: Neuron = neuronSource
  def getNeuronTarget: Neuron = neuronTarget
}
