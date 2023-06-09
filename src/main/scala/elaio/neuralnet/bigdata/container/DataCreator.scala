package elaio.neuralnet.bigdata.container

import elaio.neuralnet.units.Neuron
import elaio.neuralnet.units.NeuronType

trait DataCreator {

  def create( neuronType: NeuronType.Value ): Neuron

  protected def createInput( ): Neuron
  protected def createHidden( ): Neuron
  protected def createBackpropagation( ): Neuron
}
