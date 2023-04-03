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

    //NetTrace.WriteMessage("part tensored container - build")

    val neuronDataCreatorTensored = new NeuronDataCreator

    val container = new TensoredContainer(
      7,
      5,
      neuronDataCreatorTensored,
      true,
    )
    container.init()
    //NetTrace.WriteMessage("part tensored container - trigger")

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
        NetTrace.WriteMessage("init level begin")
        for( inputNode <- container.inputNodes) {
          NetTrace.WriteMessage("initValue: " + inputValue(index) + "- init: " + init)
          inputNode.init(inputValue(index), inputValue(index), tolerance)
          index = index + 1
        }
        NetTrace.WriteMessage("init level end")
      }

      index = -1
      for( outputNode <- container.outputNodes ) {
        index = index + 1
        NetTrace.WriteMessage("out index: " + (index + 1))
        var outValue : Double = outputNode.collectInConnections()
        NetTrace.WriteMessage("out test index: " + index + " -> " + outValue)
        NetTrace.WriteMessage("out test value: " + inputValue(index).toString)
        if( outValue > inputValue(index) - tolerance && outValue < inputValue(index) + tolerance ) {
          if( inDepth == true ) {
            outValues = outValues :+ outValue
            NetTrace.WriteMessage("return 1 " + outValue)
            return outValues
            NetTrace.WriteMessage("hidden")
          }
        }
        NetTrace.WriteMessage("step 1 ")
        val inValueNew: Array[Double] = Array(outValue)
        var doExit: Boolean = false
        NetTrace.WriteMessage("feedback 1")
        for( feedback <- feedbackIn(container, inValueNew, tolerance, false) ) {
          NetTrace.WriteMessage("feedback 2")
            if( feedback > inputValue(index) - tolerance && feedback < inputValue(index) + tolerance ) {
            NetTrace.WriteMessage("feedback 3")
            outValues = outValues :+ feedback
            NetTrace.WriteMessage("out added " + outValue + " - index: " + index + " - target " + inputValue(index))
          }
        }
      }
      NetTrace.WriteMessage("return 3 ")
      outValues
  } 
}
