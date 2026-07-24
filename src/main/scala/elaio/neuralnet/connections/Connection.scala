package elaio.neuralnet.connections
import scala.math.sqrt
import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache
trait Connection{
  val neuronSource: Neuron
  val neuronTarget: Neuron
  var weight: Double = scala.util.Random.nextDouble() * 2d - 1d
  def collect(pullWeight: Double, backpropagation: Boolean): Double = {
    val cachedNeuron = NeuronCollectionCache.get(neuronSource.id)
    val neuronValue =
      if (cachedNeuron != null) {
        cachedNeuron.value
      } else {
        val v = neuronSource.collectInConnections(pullWeight, backpropagation)
        NeuronCollectionCache.add(neuronSource)
        v
      }
    neuronValue * weight
  }
  def getNeuronSource: Neuron = neuronSource
  def getNeuronTarget: Neuron = neuronTarget
}
