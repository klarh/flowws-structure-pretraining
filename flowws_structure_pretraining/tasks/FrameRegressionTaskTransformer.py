import itertools

import flowws
from flowws import Argument as Arg
import numpy as np

from .internal import TaskTransformer
from flowws_keras_geometry.data.internal import ScaledMAE


@flowws.add_stage_arguments
class FrameRegressionTaskTransformer(TaskTransformer):
    """Generate training data to regress a target property on point clouds.

    This TaskTransformer operates lazily, producing data sequentially
    over all particles in a frame after recalculating the neighbor
    list for that frame.

    """

    ARGS = TaskTransformer.ARGS + [
        Arg(
            'context_label',
            None,
            str,
            help='Use this value as the key to extract labels from context dictionaries',
        ),
        Arg(
            'normalize_labels_batches',
            None,
            int,
            1024,
            help='Number of batches to sample to rescale labels with a mean of 0 and variance of 1',
        ),
        Arg(
            'x_scale', '-x', float, 1.0, help='Scale by which to divide input distances'
        ),
        Arg('seed', '-s', int, 13, help='RNG seed for data generation'),
        Arg('secondary_labels', None, [str], help='Additional context labels to load'),
        Arg(
            'scale_secondary_labels',
            None,
            bool,
            False,
            help='If True, rescale all secondary label values with fitted normalization',
        ),
    ]

    def run(self, scope, storage):
        super().run(scope, storage)

        self.per_bond_label = False

        self.y_shift = 0.0
        self.y_scale = 1.0

        sample_batches = self.arguments['normalize_labels_batches']
        y_measurements = []
        for (_, y) in itertools.islice(scope['train_generator'], sample_batches):
            if isinstance(y, list):
                y = y[0]
            y_measurements.append(y)
        self.y_shift = np.mean(y_measurements)
        self.y_scale = np.std(y_measurements)

        metrics = scope.setdefault('metrics', [])
        if not any(isinstance(m, ScaledMAE) for m in metrics):
            scaled_mae = ScaledMAE(self.y_scale, name='scaled_mae')
            metrics.append(scaled_mae)

        scope['normalize_label_mean'] = self.y_shift
        scope['normalize_label_std'] = self.y_scale
        scope['x_scale'] = self.arguments['x_scale']

    def transform(self, gen, evaluate, seed):
        x_scale = self.arguments['x_scale']
        y_label = self.arguments['context_label']
        secondary_labels = self.arguments.get('secondary_labels', [])
        scale_secondary_labels = self.arguments['scale_secondary_labels']

        for (rijs, tijs, weights), context in gen:
            y = (context[y_label] - self.y_shift) / self.y_scale
            y = np.array([y])
            x = rijs
            x /= x_scale

            if self.use_weights:
                x = (x[None], tijs[None], weights[None])
            else:
                x = (x[None], tijs[None])

            if secondary_labels:
                secondary_ys = [
                    np.array([context[label]]) for label in secondary_labels
                ]
                if scale_secondary_labels:
                    for i in range(len(secondary_ys)):
                        secondary_ys[i] /= self.y_scale
                y = [y] + secondary_ys

            yield x, y
