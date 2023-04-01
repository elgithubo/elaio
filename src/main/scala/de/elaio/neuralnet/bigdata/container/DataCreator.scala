package de.elaio.neuralnet.bigdata.container

import de.elaio.neuralnet.units.Neuron
import de.elaio.neuralnet.units.NeuronType

trait DataCreator {

  def create( neuronType: NeuronType.Value ): Neuron

  protected def createInput( ): Neuron
  protected def createHidden( ): Neuron
  protected def createOutput( ): Neuron
}
