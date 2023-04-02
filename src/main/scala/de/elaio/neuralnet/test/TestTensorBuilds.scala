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

    var inputValue : Double = 1.0d
    feedbackIn(container, inputValue)

    NetTrace.WriteMessage("end of test run")
  }

  def feedbackIn(container: TensoredContainer, inputValue: Double): Double = {
      var outValue : Double = 0.0d
      container.inputNodes.foreach( _.init(inputValue))
      for( outputNode <- container.outputNodes ) {
        outValue = outputNode.collectInConnections()
        if( outValue < 0.9d || outValue > 193.0d ) {
          NetTrace.WriteMessage("sub trigger: " + outValue)
          outValue = feedbackIn(container, outValue)
        }
      }
      NetTrace.WriteMessage("outValue: " + outValue ) 
      outValue
  } 
}
