from flowws import try_to_import

AutoencoderVisualizer = try_to_import(
    '.AutoencoderVisualizer', 'AutoencoderVisualizer', __name__
)
BondDenoisingVisualizer = try_to_import(
    '.BondDenoisingVisualizer', 'BondDenoisingVisualizer', __name__
)
ClassifierPlotter = try_to_import('.ClassifierPlotter', 'ClassifierPlotter', __name__)
EmbeddingDistance = try_to_import('.EmbeddingDistance', 'EmbeddingDistance', __name__)
EmbeddingDistanceTrajectory = try_to_import(
    '.EmbeddingDistanceTrajectory', 'EmbeddingDistanceTrajectory', __name__
)
EmbeddingPlotter = try_to_import('.EmbeddingPlotter', 'EmbeddingPlotter', __name__)
EvaluateEmbedding = try_to_import('.EvaluateEmbedding', 'EvaluateEmbedding', __name__)
NoisyBondVisualizer = try_to_import(
    '.NoisyBondVisualizer', 'NoisyBondVisualizer', __name__
)
PCAEmbedding = try_to_import('.PCAEmbedding', 'PCAEmbedding', __name__)
RegressorPlotter = try_to_import('.RegressorPlotter', 'RegressorPlotter', __name__)
ShiftIdentificationVisualizer = try_to_import(
    '.ShiftIdentificationVisualizer', 'ShiftIdentificationVisualizer', __name__
)
UMAPEmbedding = try_to_import('.UMAPEmbedding', 'UMAPEmbedding', __name__)
