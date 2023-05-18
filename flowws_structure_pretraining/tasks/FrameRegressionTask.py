import flowws
from flowws import Argument as Arg
import numpy as np

from .FrameClassificationTask import FrameClassificationTask


@flowws.add_stage_arguments
class FrameRegressionTask(FrameClassificationTask):
    """Generate training data to identify from which frame a sample came, as a continuous quantity"""

    ARGS = FrameClassificationTask.ARGS + [
        Arg(
            'context_label',
            None,
            str,
            help='If given, use this value as the key to extract labels from context dictionaries',
        ),
        Arg(
            'normalize_labels',
            None,
            bool,
            False,
            help='If True, normalize labels to have a mean of 0 and variance of 1',
        ),
    ]

    def run(self, scope, storage):
        super().run(scope, storage)
        if 'context_label' in self.arguments:
            key = self.arguments['context_label']
            ys = [ctx[key] for ctx in scope['x_contexts']]
        else:
            class_remap = np.linspace(0, 1, len(scope['label_remap']))
            ys = class_remap[scope['y_train'][..., 0]]

        if self.arguments['normalize_labels']:
            from flowws_keras_geometry.data.internal import ScaledMAE

            mu, sigma = np.mean(ys), np.std(ys)
            ys = (np.array(ys) - mu) / sigma
            if 'validation_data' in scope:
                scope['validation_data'][-1] -= mu
                scope['validation_data'][-1] /= sigma
            scope['normalize_label_mean'] = mu
            scope['normalize_label_std'] = sigma

            metrics = scope.setdefault('metrics', [])
            scaled_mae = ScaledMAE(sigma, name='scaled_mae')
            metrics.append(scaled_mae)

        scope['y_train'] = np.asarray(ys)
        scope['loss'] = 'mse'
        scope['metrics'].remove('accuracy')
        scope['metrics'].append('mean_absolute_error')
