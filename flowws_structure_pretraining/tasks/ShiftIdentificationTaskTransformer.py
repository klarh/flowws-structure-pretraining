import flowws
from flowws import Argument as Arg
import numpy as np

from .internal import TaskTransformer


@flowws.add_stage_arguments
class ShiftIdentificationTaskTransformer(TaskTransformer):
    """Shift input point clouds in 3D and predict the direction of the shift.

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
        Arg('scale', None, float, 1e-2, help='Magnitude of point cloud shift'),
    ]

    def run(self, scope, storage):
        super().run(scope, storage)

        self.per_bond_label = False

        scope['x_scale'] = self.arguments['x_scale']
        scope['loss'] = self.arguments['loss']
        scope.setdefault('metrics', []).append('mae')

    def transform(self, gen, evaluate, seed):
        rng = np.random.default_rng(seed)
        x_scale = self.arguments['x_scale']

        for (rijs, tijs, weights), context in gen:
            y = rng.normal(scale=self.arguments['scale'], size=(3,))
            if evaluate:
                y[:] = 0
            x = rijs + y[None, :]

            x /= x_scale
            y /= x_scale

            if self.use_weights:
                x = (x[None], tijs[None], weights[None])
            else:
                x = (x[None], tijs[None])

            yield x, y[None]
