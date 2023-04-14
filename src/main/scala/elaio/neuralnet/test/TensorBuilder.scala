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
      10,
      6,
      neuronDataCreatorTensored,
      true,
    )
    container.init()

    val outValues: Array[Double] = feedbackIn(container, Array(6d, 5d, 4d, 3d, 2d, 1d), 0.5d, true)
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
          
    var doContinue: Boolean = false           
    for(inputValue <- inputValues) {
      var currentTarget: Double = -1d
      index = index + 1
      doContinue = false   
      if( doContinue == false) {
        if(init == true) {
          for( inputNode <- container.inputNodes) {
              var initValue: Double = inputNode.target - inputNode.value + inputValue
              inputNode.init(initValue, inputValue, tolerance)
          }
        }          
        for(backpropagationNode <- container.backpropagationNodes) {
          outValue = backpropagationNode.collectInConnections(inputValue)
          NetTrace.WriteMessage( "received outvalue 1: " + outValue + " - searched: " +  inputValue)
          if( outValue > inputValue - tolerance && outValue < inputValue + tolerance ) {
            outValues = outValues :+ outValue
            NetTrace.WriteMessage( "found outvalue 1: " + outValue + " searched: " +  inputValue)
            doContinue = true
          }
        }
      }
    }
    outValues
  } 
}
