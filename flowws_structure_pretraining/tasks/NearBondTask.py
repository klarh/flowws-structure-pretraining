from .internal import pad, process_frame, EnvironmentGenerator

import flowws
from flowws import Argument as Arg
import numpy as np


@flowws.add_stage_arguments
class NearBondTask(flowws.Stage):
    """Generate training data to predict one of the nearest neighbor
    vectors from all other neighbors, where the chosen bond is selected
    based on a percentile distance criterion."""

    ARGS = [
        Arg(
            'x_scale', '-x', float, 2.0, help='Scale by which to divide input distances'
        ),
        Arg('seed', '-s', int, 13, help='RNG seed for data generation'),
        Arg('batch_size', '-b', int, 32, help='Batch size to use'),
        Arg('loss', '-l', str, 'mse', help='Loss to use when training'),
        Arg('subsample', None, float, help='Take only the given fraction of data'),
        Arg(
            'validation_split',
            '-v',
            float,
            0.3,
            help='Fraction of data to be used for validation',
        ),
        Arg('shuffle', None, bool, True, help='If True, shuffle data'),
        Arg(
            'nearest_fraction',
            None,
            float,
            0.5,
            help='Select the target bond from the given `nearest_fraction` of bonds',
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
            env_gen,
            self.arguments['seed'],
            evaluate=False,
            subsample=train_sample,
        )
        scope['validation_generator'] = self.batch_generator(
            env_gen,
            self.arguments['seed'],
            evaluate=False,
            subsample=val_sample,
        )
        scope['test_generator'] = self.batch_generator(
            env_gen, self.arguments['seed'] + 2
        )
        scope['data_generator'] = self.batch_generator(
            env_gen,
            0,
            evaluate=True,
            subsample=self.arguments.get('subsample', 1),
        )
        scope['x_scale'] = self.arguments['x_scale']
        scope['loss'] = self.arguments['loss']
        scope.setdefault('metrics', []).append('mae')

    def batch_generator(
        self,
        env_gen,
        seed=13,
        round_neighbors=4,
        evaluate=False,
        subsample=None,
    ):
        env_gen = env_gen.sample(seed, not evaluate, subsample)
        rng = np.random.default_rng(seed + 1)
        x_scale = self.arguments['x_scale']
        batch_size = self.arguments['batch_size']
        done = False

        while not done:
            rs, vs, ys, ws, ctxs = [], [], [], [], []
            max_size = 0
            while len(rs) < batch_size:
                try:
                    ((r, v, w), context) = next(env_gen)
                except StopIteration:
                    done = True
                    break
                ctxs.append(context)

                rmags = np.linalg.norm(r, axis=-1)
                rmags[np.all(v == 0, axis=-1)] += 1e9
                sortidx = np.argsort(rmags, axis=-1)
                N_eligible = int(self.arguments['nearest_fraction'] * len(sortidx))

                eligible, ineligible = sortidx[:N_eligible], sortidx[N_eligible:]
                rng.shuffle(eligible)
                selection = eligible[0]
                eligible = eligible[1:]

                neighbors = np.concatenate([eligible, ineligible])
                x = r[neighbors]
                y = r[selection]
                v = v[neighbors]
                w = w[neighbors]

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

            if not len(x[0]):
                return
            if evaluate:
                yield x, np.array(ys), ctxs
            else:
                yield x, np.array(ys)
