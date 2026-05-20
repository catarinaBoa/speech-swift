import Foundation
import MLX
import MLXNN
import MLXCommon
import AudioCommon

extension TTSWeightLoader {

    // MARK: - Speech Tokenizer Encoder Weight Loading

    public static func loadSpeechTokenizerEncoderWeights(
        into encoder: SpeechTokenizerEncoder,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)

        print("Found \(allWeights.count) speech tokenizer weights total (encoder load)")

        try requireWeights([
            "encoder.encoder.layers.0.conv.weight",
            "encoder.encoder.layers.14.conv.weight",
            "encoder.downsample.conv.weight",
            "encoder.encoder_transformer.layers.0.self_attn.q_proj.weight",
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.14.codebook.embed_sum",
        ], in: allWeights)

        loadEncoderRVQWeights(into: encoder.rvq, from: allWeights)

        CommonWeightLoader.applyConv1dWeights(
            to: encoder.inputConv.conv,
            prefix: "encoder.encoder.layers.0.conv",
            from: allWeights,
            transpose: true)

        let residualLayerIndices = [1, 4, 7, 10]
        let downsampleLayerIndices = [3, 6, 9, 12]
        for (i, block) in encoder.encoderBlocks.enumerated() {
            loadEncoderBlockWeights(
                to: block,
                residualPrefix: "encoder.encoder.layers.\(residualLayerIndices[i])",
                downsamplePrefix: "encoder.encoder.layers.\(downsampleLayerIndices[i]).conv",
                from: allWeights)
        }

        CommonWeightLoader.applyConv1dWeights(
            to: encoder.postConv.conv,
            prefix: "encoder.encoder.layers.14.conv",
            from: allWeights,
            transpose: true)
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.downsample.conv,
            prefix: "encoder.downsample.conv",
            from: allWeights,
            transpose: true)

        for (i, layer) in encoder.transformer.layers.enumerated() {
            loadEncoderTransformerLayerWeights(to: layer, index: i, from: allWeights)
        }

        print("Applied weights to Speech Tokenizer Encoder")
    }

    // MARK: - Encoder RVQ

    private static func loadEncoderRVQWeights(
        into rvq: EncoderRVQ,
        from weights: [String: MLXArray]
    ) {
        loadEncoderQuantizerCodebook(
            into: rvq.rvqFirst.quantizers[0].embedding,
            prefix: "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook",
            from: weights)
        CommonWeightLoader.applyConv1dWeights(
            to: rvq.rvqFirst.outputProj,
            prefix: "encoder.quantizer.semantic_residual_vector_quantizer.input_proj",
            from: weights,
            transpose: true)

        for i in 0..<rvq.rvqRest.numQuantizers {
            loadEncoderQuantizerCodebook(
                into: rvq.rvqRest.quantizers[i].embedding,
                prefix: "encoder.quantizer.acoustic_residual_vector_quantizer.layers.\(i).codebook",
                from: weights)
        }
        CommonWeightLoader.applyConv1dWeights(
            to: rvq.rvqRest.outputProj,
            prefix: "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj",
            from: weights,
            transpose: true)
    }

    private static func loadEncoderQuantizerCodebook(
        into embedding: Embedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        if let embed = weights["\(prefix).embed"] {
            embedding.update(parameters: ModuleParameters(values: ["weight": .value(embed)]))
            return
        }

        let usage = weights["\(prefix).cluster_usage"]
        let sum = weights["\(prefix).embedding_sum"] ?? weights["\(prefix).embed_sum"]
        if let usage, let sum {
            let eps = MLXArray(Float(1e-7))
            let clampedUsage = maximum(usage, eps).expandedDimensions(axis: -1)
            let computed = sum / clampedUsage
            embedding.update(parameters: ModuleParameters(values: ["weight": .value(computed)]))
        }
    }

    // MARK: - Encoder Block Weights

    private static func loadEncoderBlockWeights(
        to block: EncoderBlock,
        residualPrefix: String,
        downsamplePrefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyConv1dWeights(
            to: block.residualUnit.conv1.conv,
            prefix: "\(residualPrefix).block.1.conv",
            from: weights,
            transpose: true)
        CommonWeightLoader.applyConv1dWeights(
            to: block.residualUnit.conv2.conv,
            prefix: "\(residualPrefix).block.3.conv",
            from: weights,
            transpose: true)
        CommonWeightLoader.applyConv1dWeights(
            to: block.downsample.conv,
            prefix: downsamplePrefix,
            from: weights,
            transpose: true)
    }

    // MARK: - Encoder Transformer Layer Weights

    private static func loadEncoderTransformerLayerWeights(
        to layer: EncoderTransformerLayer,
        index: Int,
        from weights: [String: MLXArray]
    ) {
        let prefix = "encoder.encoder_transformer.layers.\(index)"

        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        CommonWeightLoader.applyLayerNormWeights(
            to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyLayerNormWeights(
            to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        CommonWeightLoader.applyLinearWeights(
            to: layer.fc1, prefix: "\(prefix).mlp.fc1", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.fc2, prefix: "\(prefix).mlp.fc2", from: weights)

        if let scale = weights["\(prefix).self_attn_layer_scale.scale"] {
            layer.attnLayerScale.update(parameters: ModuleParameters(values: ["scale": .value(scale.reshaped([1, 1, -1]))]))
        }
        if let scale = weights["\(prefix).mlp_layer_scale.scale"] {
            layer.mlpLayerScale.update(parameters: ModuleParameters(values: ["scale": .value(scale.reshaped([1, 1, -1]))]))
        }
    }

    private static func requireWeights(_ keys: [String], in weights: [String: MLXArray]) throws {
        let missing = keys.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw NSError(
                domain: "TTSWeightLoader",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Missing SpeechTokenizerEncoder weights: \(missing.joined(separator: ", "))"]
            )
        }
    }
}
