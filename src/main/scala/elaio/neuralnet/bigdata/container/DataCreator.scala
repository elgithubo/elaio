package elaio.neuralnet.bigdata.container

import elaio.neuralnet.units.Neuron
import elaio.neuralnet.units.NeuronType

trait DataCreator {

  def create( neuronType: NeuronType.Value ): Neuron

  protected def createInput( ): Neuron
  protected def createHiddenLeakyRelu( ): Neuron
  protected def createHiddenSquare( ): Neuron
  protected def createBackpropagation( ): Neuron
}
