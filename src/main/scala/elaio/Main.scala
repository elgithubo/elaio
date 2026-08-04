// scala v3 compatibility bootstrap - needed when compiling by hand with scalac insteam of sbt.
// compile with:
//   scalac `find src -name "*.scala"`
// and run with:
//   scala -cp . Main <Task> [<PersistenceParameters>]
// must list every package that actually exists under elaio. check against:
//   grep -h "^package" `find src -type f -name "*.scala" ! -name Main.scala` | sort -u
package elaio{
  package neuralnet{
    package activation{}
    package bigdata{
      package container{}
    }
    package connections{}
    package persistence{}
    package processing{}
    package test{}
    package trace{}
    package training{}
    package units{}
  }
}

// this class is an entry point for testing and debugging.
// it is meant to call test methods and enable in-IDE debugging.
import java.nio.file.Path
import elaio.neuralnet.persistence.PersistenceAction
import elaio.neuralnet.test.{
  AdditionTest,
  DivisionTest,
  MathTestType,
  MultiplicationTest,
  PotentialTest,
  SubtractionTest
}

object Main {
  private final case class OpSpec(
      testType: MathTestType.Value,
      persistenceAction: Option[PersistenceAction]
  )

  def main(args: Array[String]): Unit = {
    val opSpec = parseArguments(args)
    main(opSpec)
  }

  private def parseArguments(args: Array[String]): OpSpec = {
    val acceptedTestTypes = MathTestType.values.mkString(", ")

    val usage = s"Expected: <task> [--save-file <path> | --load-file <path>]"
    require(args.nonEmpty, usage)

    val testType = MathTestType.values
      .find(_.toString.equalsIgnoreCase(args.head))
      .getOrElse(throw new IllegalArgumentException(
        s"Unknown math test '${args.head}'. Expected one of: $acceptedTestTypes"
      )
    )

    val persistenceAction = args.drop(1).toList match {
      case Nil => None
      case "--save-file" :: path :: Nil => Some(PersistenceAction.Save(Path.of(path)))
      case "--load-file" :: path :: Nil => Some(PersistenceAction.Load(Path.of(path)))
      case option :: Nil if option == "--save-file" || option == "--load-file" =>
        throw new IllegalArgumentException(s"Missing path after $option. $usage")
      case option :: _ if option == "--save-file" || option == "--load-file" =>
        throw new IllegalArgumentException(s"Only one persistence option is allowed. $usage")
      case option :: _ =>
        throw new IllegalArgumentException(s"Unknown option '$option'. $usage")
    }

    OpSpec(testType, persistenceAction)
  }

  private def main(opSpec: OpSpec): Unit = opSpec.testType match {
    case MathTestType.Addition       => new AdditionTest(opSpec.persistenceAction).run()
    case MathTestType.Subtraction    => new SubtractionTest(opSpec.persistenceAction).run()
    case MathTestType.Multiplication => new MultiplicationTest(opSpec.persistenceAction).run()
    case MathTestType.Division       => new DivisionTest(opSpec.persistenceAction).run()
    case MathTestType.Potential      => new PotentialTest(opSpec.persistenceAction).run()
  }
}
