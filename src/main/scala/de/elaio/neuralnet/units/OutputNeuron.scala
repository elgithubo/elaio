package de.elaio.neuralnet.units

import de.elaio.neuralnet.trace.NetTrace


class OutputNeuron extends Neuron {
  NetTrace.WriteMessage(
    neuronSource.collectInConnections()
  )
}
