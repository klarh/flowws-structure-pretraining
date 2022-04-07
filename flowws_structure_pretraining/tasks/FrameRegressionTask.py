import flowws
from flowws import Argument as Arg
import numpy as np

from .FrameClassificationTask import FrameClassificationTask


@flowws.add_stage_arguments
class FrameRegressionTask(FrameClassificationTask):
    """Generate training data to identify from which frame a sample came, as a continuous quantity"""

    ARGS = [
        Arg(
            'x_scale', '-x', float, 2.0, help='Scale by which to divide input distances'
        ),
        Arg('seed', '-s', int, 13, help='RNG seed for data generation'),
        Arg('subsample', None, float, help='Take only the given fraction of data'),
        Arg('shuffle', None, bool, True, help='If True, shuffle data'),
        Arg(
            'per_cloud',
            '-p',
            bool,
            False,
            help='If True, classify clouds rather than individual bonds',
        ),
    ]

    def run(self, scope, storage):
        super().run(scope, storage)
        class_remap = np.linspace(0, 1, len(scope['label_remap']))
        scope['y_train'] = class_remap[scope['y_train'][..., 0]]
        scope['loss'] = 'mse'
        scope['metrics'].remove('accuracy')
        scope['metrics'].append('mean_absolute_error')
