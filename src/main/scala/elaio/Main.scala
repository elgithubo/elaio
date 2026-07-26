// scala v3 compatibility bootstrap - needed when compiling by hand with scalac insteam of sbt.
// compile with: 
//   scalac `find src -name "*.scala"`
// and run with:
//   scala -cp . Main
// must list every package that actually exists under elaio. check against:
//   grep -h "^package" `find src -type f -name "*.scala" ! -name Main.scala` | sort -u
package elaio{
  package neuralnet{
    package activation{}
    package bigdata{
      package container{}
    }
    package connections{}
    package processing{}
    package test{}
    package trace{}
    package training{}
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