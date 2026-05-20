import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon
import AudioCommon

// MARK: - Qwen3-TTS Tokenizer V2 Encoder

public class EncoderResidualUnit: Module {
    @ModuleInfo var conv1: CausalConv1d
    @ModuleInfo var conv2: CausalConv1d

    public init(dim: Int) {
        self._conv1.wrappedValue = CausalConv1d(
            inputChannels: dim, outputChannels: dim / 2,
            kernelSize: 3)
        self._conv2.wrappedValue = CausalConv1d(
            inputChannels: dim / 2, outputChannels: dim,
            kernelSize: 1)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = gelu(x)
        h = conv1(h)
        h = gelu(h)
        h = conv2(h)
        return h + x
    }
}

public class EncoderBlock: Module {
    @ModuleInfo var residualUnit: EncoderResidualUnit
    @ModuleInfo var downsample: CausalConv1d

    public init(inputDim: Int, outputDim: Int, stride: Int) {
        self._residualUnit.wrappedValue = EncoderResidualUnit(dim: inputDim)
        self._downsample.wrappedValue = CausalConv1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: stride * 2, stride: stride)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(residualUnit(x))
    }
}

public class EncoderTransformerAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear

    let rope: MLXNN.RoPE

    public init(hiddenSize: Int = 512, numHeads: Int = 8, headDim: Int = 64, ropeTheta: Float = 10000.0) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: ropeTheta)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        q = rope(q)
        k = rope(k)

        let merged = SDPA.attendAndMerge(
            qHeads: q, kHeads: k, vHeads: v,
            scale: scale, mask: attentionMask)
        return oProj(merged)
    }
}

public class EncoderTransformerLayer: Module {
    @ModuleInfo var selfAttn: EncoderTransformerAttention
    @ModuleInfo var inputLayerNorm: LayerNorm
    @ModuleInfo var postAttentionLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var attnLayerScale: LayerScale
    @ModuleInfo var mlpLayerScale: LayerScale

    public init(hiddenSize: Int = 512, intermediateSize: Int = 2048) {
        self._selfAttn.wrappedValue = EncoderTransformerAttention(hiddenSize: hiddenSize)
        self._inputLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        self._postAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        self._fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self._attnLayerScale.wrappedValue = LayerScale(channels: hiddenSize)
        self._mlpLayerScale.wrappedValue = LayerScale(channels: hiddenSize)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var h = inputLayerNorm(x)
        h = selfAttn(h, attentionMask: attentionMask)
        h = x + attnLayerScale(h)

        let residual = h
        h = postAttentionLayerNorm(h)
        h = fc1(h)
        h = gelu(h)
        h = fc2(h)
        return residual + mlpLayerScale(h)
    }
}

public class EncoderTransformer: Module {
    @ModuleInfo var layers: [EncoderTransformerLayer]

    public init(config: SpeechTokenizerDecoderConfig) {
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            EncoderTransformerLayer()
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.dim(1)
        let mask: MLXArray?
        if seqLen > 1 {
            let rows = MLXArray(0..<Int32(seqLen)).expandedDimensions(axis: 1)
            let cols = MLXArray(0..<Int32(seqLen)).expandedDimensions(axis: 0)
            mask = MLX.where(cols .> rows, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
                .asType(x.dtype)
        } else {
            mask = nil
        }

        var h = x
        for layer in layers {
            h = layer(h, attentionMask: mask)
        }
        return h
    }
}

public class EncoderRVQ: Module {
    @ModuleInfo var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo var rvqRest: ResidualVectorQuantizer

    public init(config: SpeechTokenizerDecoderConfig) {
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: 1,
            codebookSize: config.semanticCodebookSize,
            codebookDim: config.codebookDim,
            outputDim: config.hiddenSize)
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: config.numQuantizers - 1,
            codebookSize: config.acousticCodebookSize,
            codebookDim: config.codebookDim,
            outputDim: config.hiddenSize)

        super.init()
    }

    public func encode(_ h: MLXArray) -> MLXArray {
        let firstCodes = rvqFirst.encode(h)
        let restCodes = rvqRest.encode(h)
        return concatenated([firstCodes, restCodes], axis: 1)
    }
}

public class SpeechTokenizerEncoder: Module {
    public let config: SpeechTokenizerDecoderConfig

    @ModuleInfo var inputConv: CausalConv1d
    @ModuleInfo var encoderBlocks: [EncoderBlock]
    @ModuleInfo var postConv: CausalConv1d
    @ModuleInfo var downsample: CausalConv1d
    @ModuleInfo var transformer: EncoderTransformer
    @ModuleInfo var rvq: EncoderRVQ

    public init(config: SpeechTokenizerDecoderConfig) {
        self.config = config

        self._inputConv.wrappedValue = CausalConv1d(
            inputChannels: 1, outputChannels: 64,
            kernelSize: 7)
        self._encoderBlocks.wrappedValue = [
            EncoderBlock(inputDim: 64, outputDim: 128, stride: 4),
            EncoderBlock(inputDim: 128, outputDim: 256, stride: 5),
            EncoderBlock(inputDim: 256, outputDim: 512, stride: 6),
            EncoderBlock(inputDim: 512, outputDim: 1024, stride: 8),
        ]
        self._postConv.wrappedValue = CausalConv1d(
            inputChannels: 1024, outputChannels: config.hiddenSize,
            kernelSize: 3)
        self._downsample.wrappedValue = CausalConv1d(
            inputChannels: config.hiddenSize, outputChannels: config.hiddenSize,
            kernelSize: 4, stride: 2, bias: false)
        self._transformer.wrappedValue = EncoderTransformer(config: config)
        self._rvq.wrappedValue = EncoderRVQ(config: config)

        super.init()
    }

    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var h = inputConv(audio)
        for block in encoderBlocks {
            h = block(h)
        }
        h = postConv(h)
        h = downsample(h)
        h = transformer(h)
        return rvq.encode(h)
    }

    public func encode(samples: [Float]) -> MLXArray {
        let audio = MLXArray(samples).reshaped([1, samples.count, 1])
        return callAsFunction(audio)
    }
}
