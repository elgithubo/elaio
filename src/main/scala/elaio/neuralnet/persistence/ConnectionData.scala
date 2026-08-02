package elaio.neuralnet.persistence

final case class ConnectionData(
    id: Long,
    sourceId: Long,
    targetId: Long,
    weight: Double
)
