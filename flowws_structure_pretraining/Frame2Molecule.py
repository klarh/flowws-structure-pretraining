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
        Arg(
            'normalize_labels_context_key',
            None,
            str,
            help='If given, only contexts with the given key set to True will be '
            'used for label normalization',
        ),
    ]

    def run(self, scope, storage):
        self.context_label = self.arguments['context_label']
        self.max_types = scope.get('max_types', 1)

        xs, ts, ys, ctxs = [], [], [], []
        max_size = 0
        for frame in scope['loaded_frames']:
            (x, t, y) = self.encode(frame)
            xs.append(x)
            ts.append(t)
            ys.append(y)
            ctxs.append(frame.context)
            max_size = max(max_size, len(x))

        print('Molecule max_size:', max_size)

        xs_array = np.zeros((len(xs), max_size, 3))
        ts_array = np.zeros((len(xs), max_size, self.max_types))
        ys_array = np.zeros((len(xs), 1))
        ctx_array = np.array(ctxs, dtype=object)

        for i, (x, t, y) in enumerate(zip(xs, ts, ys)):
            xs_array[i, : len(x)] = x
            ts_array[i, : len(x)] = t
            ys_array[i] = y

        if self.arguments['normalize_labels']:
            from flowws_keras_geometry.data.internal import ScaledMAE

            if 'normalize_label_mean' in scope:
                mu = scope['normalize_label_mean']
                sigma = scope['normalize_label_std']
            elif 'normalize_labels_context_key' in self.arguments:
                key = self.arguments['normalize_labels_context_key']
                filt = [bool(ctx.get(key, False)) for ctx in ctx_array]
                filtered_ys = ys_array[filt]
                mu, sigma = np.mean(filtered_ys), np.std(filtered_ys)
            else:
                mu, sigma = np.mean(ys_array), np.std(ys_array)

            ys_array -= mu
            ys_array /= sigma
            scope['normalize_label_mean'] = mu
            scope['normalize_label_std'] = sigma

            metrics = scope.setdefault('metrics', [])
            if not any(isinstance(m, ScaledMAE) for m in metrics):
                scaled_mae = ScaledMAE(sigma, name='scaled_mae')
                metrics.append(scaled_mae)

        scope['x_train'] = (xs_array, ts_array)
        scope['y_train'] = ys_array
        scope['x_contexts'] = ctx_array
        scope['per_molecule'] = True
        if 'mean_absolute_error' not in scope.setdefault('metrics', []):
            scope['metrics'].append('mean_absolute_error')

    def encode(self, frame):
        y = frame.context[self.context_label]
        t = np.eye(self.max_types)[frame.types]
        return (frame.positions, t, y)
