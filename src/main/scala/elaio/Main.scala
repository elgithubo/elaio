// scala v3 compatibility bootstrap - needed when compiling by hand with scalac insteam of sbt.
// compile with:
//   scalac `find src -name "*.scala"`
// and run with:
//   scala -cp . Main <Task>
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

// this class is an entry point for testing and debugging.
// it is meant to call test methods and enable in-IDE debugging.
import elaio.neuralnet.test.{
  AdditionTest,
  DivisionTest,
  MathTestType,
  MultiplicationTest,
  PotentialTest,
  SubtractionTest
}

object Main {
  def main(args: Array[String]): Unit = {
    val accepted = MathTestType.values.mkString(", ")
    require(args.length == 1, s"Expected one argument: $accepted")

    val testType = MathTestType.values
      .find(_.toString.equalsIgnoreCase(args.head))
      .getOrElse(throw new IllegalArgumentException(
        s"Unknown math test '${args.head}'. Expected one of: $accepted"
      )
    )

    main(testType)
  }

  def main(testType: MathTestType.Value): Unit = testType match {
    case MathTestType.Addition       => AdditionTest.run()
    case MathTestType.Subtraction    => SubtractionTest.run()
    case MathTestType.Multiplication => MultiplicationTest.run()
    case MathTestType.Division       => DivisionTest.run()
    case MathTestType.Potential      => PotentialTest.run()
  }
}
