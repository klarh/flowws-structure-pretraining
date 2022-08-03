from .internal import pad, process_frame, EnvironmentGenerator

import flowws
from flowws import Argument as Arg
import numpy as np


@flowws.add_stage_arguments
class ShiftIdentificationTask(flowws.Stage):
    """Shift input point clouds in 3D and predict the direction of the shift"""

    ARGS = [
        Arg(
            'x_scale', '-x', float, 2.0, help='Scale by which to divide input distances'
        ),
        Arg('seed', '-s', int, 13, help='RNG seed for data generation'),
        Arg('batch_size', '-b', int, 32, help='Batch size to use'),
        Arg('loss', '-l', str, 'mse', help='Loss to use when training'),
        Arg('scale', None, float, 1e-2, help='Magnitude of point cloud shift'),
        Arg(
            'subsample',
            None,
            float,
            help='Take only the given fraction of data for evaluation',
        ),
        Arg(
            'validation_split',
            '-v',
            float,
            0.3,
            help='Fraction of data to be used for validation',
        ),
    ]

    def run(self, scope, storage):
        self.use_weights = scope.get('use_bond_weights', False)
        max_types = scope['max_types']
        self.type_dim = 2 * max_types

        nlist_generator = scope['nlist_generator']
        frames = []
        for frame in scope['loaded_frames']:
            frame = process_frame(frame, nlist_generator, max_types)
            frames.append(frame)

        env_gen = EnvironmentGenerator(frames)

        train_sample = np.array([self.arguments['validation_split'], 1.0])
        val_sample = np.array([0, self.arguments['validation_split']])

        if 'subsample' in self.arguments:
            train_sample[1] = (
                train_sample[0] + np.diff(train_sample) * self.arguments['subsample']
            )
            val_sample[1] = (
                val_sample[0] + np.diff(val_sample) * self.arguments['subsample']
            )

        for key in ['x_train', 'y_train', 'train_generator', 'validation_generator']:
            scope.pop(key, None)
        scope['train_generator'] = self.batch_generator(
            env_gen, self.arguments['seed'], evaluate=False, subsample=train_sample
        )
        scope['validation_generator'] = self.batch_generator(
            env_gen, self.arguments['seed'], evaluate=False, subsample=val_sample
        )
        scope['test_generator'] = self.batch_generator(
            env_gen, self.arguments['seed'] + 2
        )
        scope['data_generator'] = self.batch_generator(
            env_gen, 0, evaluate=True, subsample=self.arguments.get('subsample', 1)
        )
        scope['x_scale'] = self.arguments['x_scale']
        scope['loss'] = self.arguments['loss']
        scope.setdefault('metrics', []).append('mae')

    def batch_generator(
        self, env_gen, seed=13, round_neighbors=4, evaluate=False, subsample=None
    ):
        env_gen = env_gen.sample(seed, not evaluate, subsample)
        rng = np.random.default_rng(seed + 1)
        x_scale = self.arguments['x_scale']
        batch_size = self.arguments['batch_size']

        while True:
            rs, vs, ys, ws, ctxs = [], [], [], [], []
            max_size = 0
            while len(rs) < batch_size:
                try:
                    ((r, v, w), context) = next(env_gen)
                except StopIteration:
                    return
                ctxs.append(context)

                y = rng.normal(scale=self.arguments['scale'], size=(3,))
                if evaluate:
                    y[:] = 0
                x = r + y[None, :]

                x /= x_scale
                y /= x_scale

                max_size = max(max_size, len(r))
                rs.append(x)
                vs.append(v)
                ys.append(y)
                ws.append(w)

            max_size = (
                (max_size + round_neighbors - 1) // round_neighbors * round_neighbors
            )
            x = (
                pad(rs, max_size, 3),
                pad(vs, max_size, self.type_dim),
                pad(ws, max_size, None),
            )
            if not self.use_weights:
                x = x[:-1]

            ys = np.array(ys)

            if evaluate:
                yield x, ys, ctxs
            else:
                yield x, ys
