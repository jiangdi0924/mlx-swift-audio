// Multi-layer unidirectional LSTM for Chatterbox VoiceEncoder.
// Uses optimized operations (addMM, split) matching MLX's built-in patterns.
//
// TODO: Once https://github.com/ml-explore/mlx-swift/pull/312 is merged,
// consider replacing this with MLXNN.LSTM (bias will be loadable via @ParameterInfo).

import Foundation
import MLX
import MLXNN

// MARK: - OptimizedLSTMCell

/// Single LSTM layer with optimized operations and loadable bias parameter
///
/// Uses the same optimizations as MLX's built-in LSTM:
/// - `addMM` for fused bias + matmul
/// - `split` for efficient gate extraction
class OptimizedLSTMCell: Module {
  let hiddenSize: Int

  // MLX format weight names (matches built-in LSTM)
  @ParameterInfo(key: "Wx") var Wx: MLXArray
  @ParameterInfo(key: "Wh") var Wh: MLXArray
  @ParameterInfo(key: "bias") var bias: MLXArray

  init(inputSize: Int, hiddenSize: Int) {
    self.hiddenSize = hiddenSize
    _Wx.wrappedValue = MLXArray.zeros([4 * hiddenSize, inputSize])
    _Wh.wrappedValue = MLXArray.zeros([4 * hiddenSize, hiddenSize])
    _bias.wrappedValue = MLXArray.zeros([4 * hiddenSize])
  }

  /// Process sequence through LSTM layer using optimized operations
  func callAsFunction(
    _ x: MLXArray,
    hidden: MLXArray? = nil,
    cell: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    // Fused bias + matmul for input projection (like built-in LSTM)
    var xProj = addMM(bias, x, Wx.T)

    var h: MLXArray! = hidden
    var c: MLXArray! = cell
    var allHidden = [MLXArray]()
    var allCell = [MLXArray]()

    let seqLen = xProj.dim(-2)
    for index in 0 ..< seqLen {
      var ifgo = xProj[.ellipsis, index, 0...]
      if h != nil {
        ifgo = addMM(ifgo, h, Wh.T)
      }

      // Use split for efficient gate extraction (like built-in LSTM)
      let pieces = split(ifgo, parts: 4, axis: -1)

      let i = sigmoid(pieces[0])
      let f = sigmoid(pieces[1])
      let g = tanh(pieces[2])
      let o = sigmoid(pieces[3])

      if c != nil {
        c = f * c + i * g
      } else {
        c = i * g
      }
      h = o * tanh(c)

      allCell.append(c)
      allHidden.append(h)
    }

    return (stacked(allHidden, axis: -2), stacked(allCell, axis: -2))
  }
}

// MARK: - ChatterboxLSTM

/// Multi-layer unidirectional LSTM with optimized operations
///
/// Weight naming follows MLX convention (after sanitization from PyTorch):
/// - layers.0.Wx, layers.0.Wh, layers.0.bias for layer 0
/// - layers.1.Wx, layers.1.Wh, layers.1.bias for layer 1
/// - layers.2.Wx, layers.2.Wh, layers.2.bias for layer 2
class ChatterboxLSTM: Module {
  let inputSize: Int
  let hiddenSize: Int
  let numLayers: Int

  @ModuleInfo(key: "layers") var layers: [OptimizedLSTMCell]

  init(inputSize: Int, hiddenSize: Int, numLayers: Int = 3) {
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.numLayers = numLayers

    // Create layers: first layer takes inputSize, rest take hiddenSize
    var layerArray: [OptimizedLSTMCell] = []
    for i in 0 ..< numLayers {
      let layerInputSize = i == 0 ? inputSize : hiddenSize
      layerArray.append(OptimizedLSTMCell(inputSize: layerInputSize, hiddenSize: hiddenSize))
    }
    _layers.wrappedValue = layerArray
  }

  /// Process sequence through all LSTM layers
  func callAsFunction(
    _ x: MLXArray,
    hidden: (MLXArray, MLXArray)? = nil,
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    var hList: [MLXArray?]
    var cList: [MLXArray?]

    if let (h0, c0) = hidden {
      hList = (0 ..< numLayers).map { i in h0[i] }
      cList = (0 ..< numLayers).map { i in c0[i] }
    } else {
      hList = [MLXArray?](repeating: nil, count: numLayers)
      cList = [MLXArray?](repeating: nil, count: numLayers)
    }

    var currentOutput = x
    var finalHidden: [MLXArray] = []
    var finalCell: [MLXArray] = []

    for i in 0 ..< numLayers {
      let (allH, allC) = layers[i](currentOutput, hidden: hList[i], cell: cList[i])
      currentOutput = allH

      // Extract final timestep for h and c
      let seqLen = allH.dim(-2)
      finalHidden.append(allH[.ellipsis, seqLen - 1, 0...])
      finalCell.append(allC[.ellipsis, seqLen - 1, 0...])
    }

    let hN = MLX.stacked(finalHidden, axis: 0)
    let cN = MLX.stacked(finalCell, axis: 0)

    return (currentOutput, (hN, cN))
  }
}
