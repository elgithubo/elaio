package de.elaio.neuralnet.test

import de.elaio.neuralnet.bigdata.container.TensoredContainer
import de.elaio.neuralnet.trace.NetTrace
import de.elaio.neuralnet.units.NeuronDataCreator
import de.elaio.neuralnet.processing.NeuronCollectionCache

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
    )
    container.init()
    NetTrace.WriteMessage("part tensored container - trigger")

    var inputValue : Double = 1.1d
    feedbackIn(container, inputValue)

    NetTrace.WriteMessage("end of test run")
  }

  def feedbackIn(container: TensoredContainer, inputValue: Double): Array[Double] = {
      var outValues: Array[Double] = Array.ofDim[Double](0)
      container.inputNodes.foreach( _.init(inputValue))
      for( outputNode <- container.outputNodes ) {
        var outValue : Double = outputNode.collectInConnections()
        if( outValue < 0.9d || outValue > 1.1d ) {
          NetTrace.WriteMessage("sub trigger: " + outValue)
          NeuronCollectionCache.clear()
          outValues +: feedbackIn(container, outValue)
        }
      }
      //for( outValue <- OutValues ) {
        //NetTrace.WriteMessage("outValue: " + outValue ) 
      //}
      outValues
  } 
}
