name := "elaio"
version := "0.2"
scalaVersion := "3.8.4"

scalacOptions += "-Wunused:imports"

// test with 'sbt Test/executeTests'
libraryDependencies += "org.scalameta" %% "munit" % "1.3.5" % Test
