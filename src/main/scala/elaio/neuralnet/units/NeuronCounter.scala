package elaio.neuralnet.units

object NeuronCounter {
  private var counter = 0L

  def getNext(): Long = {
    counter = counter + 1L
    counter
  }

  def current: Long = counter
}
