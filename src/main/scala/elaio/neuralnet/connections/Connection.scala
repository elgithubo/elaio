package elaio.neuralnet.connections
import scala.math.sqrt
import elaio.neuralnet.units.Neuron
import elaio.neuralnet.processing.NeuronCollectionCache
trait Connection{
  val neuronSource: Neuron
  val neuronTarget: Neuron

  // initialize weight based on a rendom number for now
  var weight: Double = scala.util.Random.nextDouble() * 2d - 1d

  def collect(): Double = {
    val cachedNeuron = NeuronCollectionCache.get(neuronSource.id)
    val neuronValue =
      if (cachedNeuron != null) {
        cachedNeuron.value
      } else {
        val v = neuronSource.collectInConnections()
        NeuronCollectionCache.add(neuronSource)
        v
      }
    neuronValue * weight
  }

  def getNeuronSource: Neuron = neuronSource

  def getNeuronTarget: Neuron = neuronTarget
}
