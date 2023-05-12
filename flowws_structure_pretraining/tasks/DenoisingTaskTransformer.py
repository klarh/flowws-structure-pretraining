import flowws
from flowws import Argument as Arg
import numpy as np
import rowan

from .internal import TaskTransformer


@flowws.add_stage_arguments
class DenoisingTaskTransformer(TaskTransformer):
    """Generate training data to reproduce noisy versions of input point clouds.

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
            'noise',
            '-n',
            float,
            0.1,
            help='Magnitude of noise to use (before dividing by x_scale)',
        ),
        Arg(
            'register',
            '-r',
            bool,
            False,
            help='If True, register point clouds after adding noise',
        ),
    ]

    def run(self, scope, storage):
        super().run(scope, storage)

        self.per_bond_label = True

        scope['x_scale'] = self.arguments['x_scale']
        scope['loss'] = self.arguments['loss']
        scope.setdefault('metrics', []).append('mae')

    def transform(self, gen, evaluate, seed):
        rng = np.random.default_rng(seed)
        register = self.arguments['register']
        x_scale = self.arguments['x_scale']

        for (rijs, tijs, weights), context in gen:
            y = rijs

            if evaluate:
                x = rijs.copy()
            else:
                noise = rng.normal(scale=self.arguments['noise'], size=rijs.shape)
                noise -= np.mean(noise, axis=0, keepdims=True)
                x = rijs + noise

                if register:
                    (R, t) = rowan.mapping.kabsch(x, y)
                    x = x @ R.T + t

            x /= x_scale
            y /= x_scale

            if self.use_weights:
                x = (x[None], tijs[None], weights[None])
            else:
                x = (x[None], tijs[None])

            yield x, y[None]
