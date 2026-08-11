package elaio.neuralnet.test

import elaio.neuralnet.persistence.PersistenceAction
import elaio.neuralnet.trace.NetTrace

// The same retrieval task as AttentionTokenTest, but with one container per token instead of one
// container for the lot. The containers share their weights, so the stack is one function applied
// to every token. Attention still runs across depths, so the stack is the only difference.
final class LayeredDepthTest(override protected val persistenceAction: Option[PersistenceAction] = None)
    extends AttentionTokenTest(persistenceAction) {
  override protected val layeredTokens = true
  override protected val dimOuter = 2
  override protected def traceAction(): Unit =
    NetTrace.WriteMessage("testing: attention with a token matrix over layered tensored containers")
}
