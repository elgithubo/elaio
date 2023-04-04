package elaio.neuralnet.test

import elaio.neuralnet.bigdata.container.TensoredContainer
import elaio.neuralnet.trace.NetTrace
import elaio.neuralnet.units.NeuronDataCreator
import elaio.neuralnet.processing.NeuronCollectionCache

object TensorBuilder {
  def run(): Unit = {
    // Enable the following line to write detailed trace messages to stdout
    NetTrace.started_(true)

    NetTrace.WriteMessage("start of test run")

    NetTrace.WriteMessage("backpropagation only working in pre-alpha mode")

    val neuronDataCreatorTensored = new NeuronDataCreator

    val container = new TensoredContainer(
      7,
      5,
      neuronDataCreatorTensored,
      true,
    )
    container.init()

    val outValues: Array[Double] = feedbackIn(container, Array(1d, 3d, 5d, 2d, 4d), 0.5d, true)
    for( outValue <- outValues ) {
      NetTrace.WriteMessage("outValue: " + outValue ) 
    }

    NetTrace.WriteMessage("end of test run")
  }

  def feedbackIn(container: TensoredContainer, inputValue: Array[Double], tolerance: Double, init: Boolean): Array[Double] = {
      var outValues: Array[Double] = Array.ofDim[Double](0)
      var outValuesCollected: Array[Double] = Array.ofDim[Double](0)
      var outValue: Double = 0
      var inDepth = false
      var index: Integer = 0

      if(inputValue.length == 1) inDepth = true 

      if( init == true ) {
        for( inputNode <- container.inputNodes) {
            inputNode.init(inputValue(index), inputValue(index), tolerance)
        }
      }

      index = -1
      var doExit: Boolean = false
      for( outputNode <- container.outputNodes ) {
        var lastOutValue: Double = 0
        if( !doExit ) {
          index = index + 1
          var outValue : Double = outputNode.collectInConnections()
          if(outValue == lastOutValue) doExit = true
          lastOutValue = outValue
          if( outValue > inputValue(index) - tolerance && outValue < inputValue(index) + tolerance ) {
            if( inDepth == true ) {
              outValues = outValues :+ outValue
              doExit = true
            }
          }
          if( !doExit ) {
            val inValueNew: Array[Double] = Array(outValue)
            for( feedback <- feedbackIn(container, inValueNew, tolerance, false) ) {
                if( feedback > inputValue(index) - tolerance && feedback < inputValue(index) + tolerance ) {
                outValues = outValues :+ feedback
              }
            }
          }
        }
      }
      outValues
  } 
}
