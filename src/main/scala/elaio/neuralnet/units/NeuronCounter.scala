package elaio.neuralnet.units

object NeuronCounter {
  private var counter: Double = 0

  def getNext(): Double = {
    counter = counter+1
    counter
  }

  def current: Double = counter
}
