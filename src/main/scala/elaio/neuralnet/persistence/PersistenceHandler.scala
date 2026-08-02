package elaio.neuralnet.persistence

import java.io.{InvalidObjectException, ObjectInputStream, ObjectOutputStream}
import java.nio.file.{Files, Path}
import scala.util.Using

final class PersistenceHandler {
  def save(stateContainer: StateContainer, path: Path): Unit =
    Using.resource(new ObjectOutputStream(Files.newOutputStream(path))) { output =>
      output.writeObject(stateContainer)
    }

  def load(path: Path): StateContainer =
    Using.resource(new ObjectInputStream(Files.newInputStream(path))) { input =>
      input.readObject() match {
        case stateContainer: StateContainer => stateContainer
        case _ => throw new InvalidObjectException("File does not contain a StateContainer")
      }
    }
}
