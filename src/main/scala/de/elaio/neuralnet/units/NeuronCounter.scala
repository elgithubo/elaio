package de.elaio.neuralnet.units

object NeuronCounter {
  private var counter: Int = 0

  def getNext(): Int = {
    counter = counter+1
    counter
  }
}
