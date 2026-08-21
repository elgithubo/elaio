name := "elaio"
version := "0.2"
scalaVersion := "3.8.4"

scalacOptions += "-Wunused:imports"

// munit supplies the framework "sbt test" discovers; suites live under src/test/scala
libraryDependencies += "org.scalameta" %% "munit" % "1.3.5" % Test
