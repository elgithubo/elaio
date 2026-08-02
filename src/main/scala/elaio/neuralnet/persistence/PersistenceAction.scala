package elaio.neuralnet.persistence

import java.nio.file.Path

enum PersistenceAction {
  case Save(file: Path)
  case Load(file: Path)
}
