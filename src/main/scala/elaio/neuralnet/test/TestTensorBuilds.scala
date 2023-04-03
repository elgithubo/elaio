package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.NeuronDataCreator
import elaio.neuralnet.processing.NeuronCollectionCache

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

    val outValues: Array[Double] = feedbackIn(container, 1d, 1d, 1d)
    for( outValue <- outValues ) {
      NetTrace.WriteMessage("outValue: " + outValue ) 
    }

    NetTrace.WriteMessage("end of test run")
  }

  def feedbackIn(container: TensoredContainer, inputValue: Double, target: Double, tolerance: Double): Array[Double] = {
      var outValues: Array[Double] = Array.ofDim[Double](0)
      var outValuesCollected: Array[Double] = Array.ofDim[Double](0)
      var outValue: Double = 0
      var hit: Boolean = true

      container.inputNodes.foreach( _.init(inputValue, target, tolerance))

      for( outputNode <- container.outputNodes ) {
        var outValue : Double = outputNode.collectInConnections()
        if( hit == true && ( outValue < target - tolerance || outValue > target + tolerance ) ) {
          hit = false
        } else {
          outValuesCollected = outValuesCollected :+ outValue
        }
      }
      
      if( hit == true ) {
        return outValuesCollected
      }

      for( outputNode <- container.outputNodes ) {
        var outValue : Double = outputNode.collectInConnections()
        for( feedback <- feedbackIn(container, outValue, target, tolerance) ) outValues = outValues :+ feedback
      }
      outValues
  } 
}
