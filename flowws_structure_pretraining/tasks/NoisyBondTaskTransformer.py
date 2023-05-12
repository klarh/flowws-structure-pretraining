import flowws
from flowws import Argument as Arg
import numpy as np

from .internal import TaskTransformer


@flowws.add_stage_arguments
class NoisyBondTaskTransformer(TaskTransformer):
    """Generate training data to identify which bonds have had random noise added.

    This TaskTransformer operates lazily, producing data sequentially
    over all particles in a frame after recalculating the neighbor
    list for that frame.

    """

    ARGS = TaskTransformer.ARGS + [
        Arg(
            'x_scale', '-x', float, 2.0, help='Scale by which to divide input distances'
        ),
        Arg('loss', '-l', str, 'mse', help='Loss to use when training'),
        Arg('seed', '-s', int, 13, help='RNG seed for data generation'),
        Arg(
            'noise_magnitude',
            '-n',
            float,
            5e-1,
            help='Magnitude of noise to add to coordinates (before dividing by x_scale)',
        ),
    ]

    def run(self, scope, storage):
        super().run(scope, storage)

        self.per_bond_label = True

        scope['x_scale'] = self.arguments['x_scale']
        if not scope.get('freeze_num_classes', False):
            scope['num_classes'] = 2
        scope['per_cloud'] = False
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope.setdefault('metrics', []).append('accuracy')

    def transform(self, gen, evaluate, seed):
        rng = np.random.default_rng(seed)
        x_scale = self.arguments['x_scale']
        noise_magnitude = self.arguments['noise_magnitude']

        for (rijs, tijs, weights), context in gen:
            x = rijs.copy()
            y = np.zeros(len(x), dtype=np.int32)
            if not evaluate:
                sel = rng.permutation(len(x))[: len(x) // 2]
                x[sel] += rng.normal(scale=noise_magnitude, size=(len(sel), 3))
                y[sel] = 1

            x /= x_scale

            if self.use_weights:
                x = (x[None], tijs[None], weights[None])
            else:
                x = (x[None], tijs[None])

            yield x, y[None]
