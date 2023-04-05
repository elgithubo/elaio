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

    val outValues: Array[Double] = feedbackIn(container, Array(6d, 3d, 5d, 2d, 4d), 0.5d, true)
    for( outValue <- outValues ) {
      NetTrace.WriteMessage("outValue: " + outValue ) 
    }

    NetTrace.WriteMessage("end of test run")
  }

  def feedbackIn(container: TensoredContainer, inputValues: Array[Double], tolerance: Double, init: Boolean): Array[Double] = {
    var outValues: Array[Double] = Array.ofDim[Double](0)
    var outValuesCollected: Array[Double] = Array.ofDim[Double](0)
    var outValue: Double = 0d
    var inDepth = false
    var index: Integer = -1

    if(inputValues.length == 1) inDepth = true 
          
    var doExit: Boolean = false           
    for(inputValue <- inputValues) {
      index = index + 1
      doExit = false   
      if(
        init == true) {
        for( inputNode <- container.inputNodes) {
            var initValue: Double = inputNode.target - inputNode.value + inputValues(index)
            inputNode.init(initValue, inputValues(index), tolerance)
        }
      }      
      var pullWeight = inputValues(index)      
      for(backpropagationNode <- container.backpropagationNodes) {
        if( !doExit ) {
          outValue = backpropagationNode.collectInConnections(pullWeight)
          if( outValue > inputValues(index) - tolerance && outValue < inputValues(index) + tolerance ) {
            outValues = outValues :+ outValue
            doExit = true
          }
          if(doExit == false) {
            val inValueNew: Array[Double] = Array(outValue)
            for(feedback <- feedbackIn(container, inValueNew, tolerance, false)) {
                if( feedback > inputValues(index) - tolerance && feedback < inputValues(index) + tolerance ) {
                outValues = outValues :+ feedback
                doExit = true
              }
            }
          }
        }
      }
    }
    outValues
  } 
}
