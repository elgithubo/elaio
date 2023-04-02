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
    //NetTrace.WriteMessage("part tensored container - trigger")

    val outValues: Array[Double] = feedbackIn(container, 1d, 1d, 1d)
    for( outValue <- outValues ) {
      //NetTrace.WriteMessage("outValue: " + outValue ) 
    }

    //NetTrace.WriteMessage("end of test run")
    //NetTrace.stop()
  }

  def feedbackIn(container: TensoredContainer, inputValue: Double, target: Double, tolerance: Double): Array[Double] = {
      //NetTrace.WriteMessage("inputValue: " + inputValue + " - tolerance: " + tolerance)
      var outValues: Array[Double] = Array.ofDim[Double](0)
      container.inputNodes.foreach( _.init(inputValue))
      for( outputNode <- container.outputNodes ) {
        var outValue : Double = outputNode.collectInConnections()
        if( outValue < target - tolerance || outValue > target + tolerance ) {
          for( feedback <- feedbackIn(container, outValue, target, tolerance) ) outValues = outValues :+ feedback
        } else {
          //NetTrace.WriteMessage("correct output: " + outValue + " - min: " + (target - tolerance) + " - max: " + (target + tolerance))
          outValues = outValues :+ outValue 
        }
      }
      outValues
  } 
}
