import flowws
from flowws import Argument as Arg
import numpy as np

from .internal import TaskTransformer


@flowws.add_stage_arguments
class AutoencoderTaskTransformer(TaskTransformer):
    """Generate training data to reproduce input point clouds.

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
    ]

    def run(self, scope, storage):
        super().run(scope, storage)

        self.per_bond_label = True

        scope['x_scale'] = self.arguments['x_scale']
        scope['loss'] = self.arguments['loss']
        scope.setdefault('metrics', []).append('mae')

    def transform(self, gen, evaluate, seed):
        x_scale = self.arguments['x_scale']

        for (rijs, tijs, weights), context in gen:
            y = rijs
            x = rijs.copy()

            x /= x_scale
            y /= x_scale

            if self.use_weights:
                x = (x[None], tijs[None], weights[None])
            else:
                x = (x[None], tijs[None])

            yield x, y[None]
