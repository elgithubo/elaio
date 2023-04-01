package de.elaio.neuralnet.test

import de.elaio.neuralnet.bigdata.container.TensoredContainer
import de.elaio.neuralnet.trace.NetTrace
import de.elaio.neuralnet.units.NeuronDataCreator

object TestTensorBuilds {
  def run(): Unit = {
    // Enable the following line to write detailed trace messages to stdout
    NetTrace.started_(true)

    NetTrace.WriteMessage("start of test run")

    NetTrace.WriteMessage("part tensored container - build")

    val neuronDataCreatorTensored = new NeuronDataCreator

    val container = new TensoredContainer(
      7,
      5,
      neuronDataCreatorTensored,
      true,
      5
    )
    container.init()
    NetTrace.WriteMessage("part tensored container - trigger")
    // for (dummy <- 0 to 5) {
    container.inputNodes.foreach( _.init(1.0f))
    for( outputNode <- container.outputNodes ) {
      val outValue = outputNode.collectInConnections()
      NetTrace.WriteMessage("outValue: " + outValue ) 
    }
    // }
    NetTrace.WriteMessage("end of test run")
  }
}
