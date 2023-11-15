from flowws import try_to_import

AutoencoderTask = try_to_import('.AutoencoderTask', 'AutoencoderTask', __name__)
AutoencoderTaskTransformer = try_to_import(
    '.AutoencoderTaskTransformer', 'AutoencoderTaskTransformer', __name__
)
DenoisingTask = try_to_import('.DenoisingTask', 'DenoisingTask', __name__)
DenoisingTaskTransformer = try_to_import(
    '.DenoisingTaskTransformer', 'DenoisingTaskTransformer', __name__
)
FrameClassificationTask = try_to_import(
    '.FrameClassificationTask', 'FrameClassificationTask', __name__
)
FrameRegressionTask = try_to_import(
    '.FrameRegressionTask', 'FrameRegressionTask', __name__
)
FrameRegressionTaskTransformer = try_to_import(
    '.FrameRegressionTaskTransformer', 'FrameRegressionTaskTransformer', __name__
)
NearBondTask = try_to_import('.NearBondTask', 'NearBondTask', __name__)
NearBondTaskTransformer = try_to_import(
    '.NearBondTaskTransformer', 'NearBondTaskTransformer', __name__
)
NearestBondTask = try_to_import('.NearestBondTask', 'NearestBondTask', __name__)
NoisyBondTask = try_to_import('.NoisyBondTask', 'NoisyBondTask', __name__)
NoisyBondTaskTransformer = try_to_import(
    '.NoisyBondTaskTransformer', 'NoisyBondTaskTransformer', __name__
)
PointGroupTask = try_to_import('.PointGroupTask', 'PointGroupTask', __name__)
ShiftIdentificationTask = try_to_import(
    '.ShiftIdentificationTask', 'ShiftIdentificationTask', __name__
)
ShiftIdentificationTaskTransformer = try_to_import(
    '.ShiftIdentificationTaskTransformer',
    'ShiftIdentificationTaskTransformer',
    __name__,
)
