package elaio.neuralnet.persistence

import java.nio.file.Path

// persistence action to be performed
enum PersistenceAction {
  case Save(file: Path)
  case Load(file: Path)
}
