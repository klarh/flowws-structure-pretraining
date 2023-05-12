import flowws
from flowws import Argument as Arg
import numpy as np

from .internal import TaskTransformer


@flowws.add_stage_arguments
class NearBondTaskTransformer(TaskTransformer):
    """Generate training data to predict one of the nearest neighbor
    vectors from all other neighbors, where the chosen bond is selected
    based on a percentile distance criterion.

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
            'nearest_fraction',
            None,
            float,
            0.5,
            help='Select the target bond from the given `nearest_fraction` of bonds',
        ),
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
            if len(rijs) < 2:
                continue

            rmags = np.linalg.norm(rijs, axis=-1)
            rmags[np.all(tijs == 0, axis=-1)] += 1e9
            sortidx = np.argsort(rmags, axis=-1)
            N_eligible = int(self.arguments['nearest_fraction'] * len(sortidx))

            eligible, ineligible = sortidx[:N_eligible], sortidx[N_eligible:]
            rng.shuffle(eligible)
            selection = eligible[0]
            eligible = eligible[1:]

            neighbors = np.concatenate([eligible, ineligible])
            x = rijs[neighbors]
            y = rijs[selection]
            v = tijs[neighbors]
            w = weights[neighbors]

            x /= x_scale
            y /= x_scale

            if self.use_weights:
                x = (x[None], v[None], w[None])
            else:
                x = (x[None], v[None])

            yield x, y[None]
