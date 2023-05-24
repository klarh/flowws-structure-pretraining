import flowws
from flowws import Argument as Arg
import numpy as np


@flowws.add_stage_arguments
class Frame2Molecule(flowws.Stage):
    """Convert an entire frame to a single dataset element"""

    ARGS = [
        Arg(
            'context_label',
            None,
            str,
            'energy',
            help='Use this value as the key to extract labels from context dictionaries',
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
        self.context_label = self.arguments['context_label']
        self.max_types = scope.get('max_types', 1)

        xs, ts, ys = [], [], []
        max_size = 0
        for frame in scope['loaded_frames']:
            (x, t, y) = self.encode(frame)
            xs.append(x)
            ts.append(t)
            ys.append(y)
            max_size = max(max_size, len(x))

        print('Molecule max_size:', max_size)

        xs_array = np.zeros((len(xs), max_size, 3))
        ts_array = np.zeros((len(xs), max_size, self.max_types))
        ys_array = np.zeros((len(xs), 1))

        for i, (x, t, y) in enumerate(zip(xs, ts, ys)):
            xs_array[i, : len(x)] = x
            ts_array[i, : len(x)] = t
            ys_array[i] = y

        if self.arguments['normalize_labels']:
            from flowws_keras_geometry.data.internal import ScaledMAE

            mu, sigma = np.mean(ys_array), np.std(ys_array)
            ys_array -= mu
            ys_array /= sigma
            scope['normalize_label_mean'] = mu
            scope['normalize_label_std'] = sigma

            scaled_mae = ScaledMAE(sigma, name='scaled_mae')
            scope.setdefault('metrics', []).append(scaled_mae)

        scope['x_train'] = (xs_array, ts_array)
        scope['y_train'] = ys_array
        scope['per_molecule'] = True
        scope.setdefault('metrics', []).append('mean_absolute_error')

    def encode(self, frame):
        y = frame.context[self.context_label]
        t = np.eye(self.max_types)[frame.types]
        return (frame.positions, t, y)
