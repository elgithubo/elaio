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
// This class is an entry point for testing and
// debugging.
// It is meant to call test methods and easily enable
// IDE debugging.
//import elaio.neuralnet.test.TestTensorBuilds
import elaio.neuralnet.test.TensorBuilder
object Main {
  def main(args: Array[String]): Unit = {
      TensorBuilder.run()
  }
}
