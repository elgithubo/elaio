package elaio.neuralnet.persistence

// to be serialized part of a connection
final case class ConnectionData(
    id: Long,
    sourceId: Long,
    targetId: Long,
    weight: Double
)
