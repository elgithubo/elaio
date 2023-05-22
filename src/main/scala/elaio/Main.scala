// v3 compatibility bootstrap
package elaio{
  package neuralnet{
    package bigdata{
      package connections{}
    }
    package processing{}
    package test{}
    package trace{}
    package units{}
  }
}
// This class is an entry point for testing and debugging.
// It is meant to call test methods and enable in-IDE debugging.
import elaio.neuralnet.test.TensorBuilder
object Main {
  def main(args: Array[String]): Unit = {
      TensorBuilder.run()
  }
}