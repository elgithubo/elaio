package elaio.neuralnet.trace

object NetTrace {
  private var _started: Boolean = false

  def started(): Boolean = {
    _started
  }

  def started_(started: Boolean): Unit =
    _started = started

  def WriteMessage(message: String, indent: BigInt = 0): Unit =
    if(_started) {
      var n = 1
      var prefix: String = ""
      while (n <= indent) {
        prefix = prefix + "."
        n += 1
      }
      //println(prefix + s"$message")a
    }
}
