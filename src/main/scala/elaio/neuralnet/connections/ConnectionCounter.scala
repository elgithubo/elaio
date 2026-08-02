package elaio.neuralnet.connections

object ConnectionCounter {
  private var _counter: Long = 0L

  def getNext(): Long = {
    _counter = _counter + 1L
    _counter
  }

  def counter: Long = _counter
}
