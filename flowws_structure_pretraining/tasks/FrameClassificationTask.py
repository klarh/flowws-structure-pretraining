import flowws
from flowws import Argument as Arg
import numpy as np

from .internal import index_frame, process_frame
from ..internal import Remap


@flowws.add_stage_arguments
class FrameClassificationTask(flowws.Stage):
    """Generate training data to identify from which frame a sample came"""

    ARGS = [
        Arg(
            'x_scale', '-x', float, 2.0, help='Scale by which to divide input distances'
        ),
        Arg('seed', '-s', int, 13, help='RNG seed for data generation'),
        Arg('subsample', None, float, help='Take only the given fraction of data'),
        Arg('shuffle', None, bool, True, help='If True, shuffle data'),
        Arg(
            'multilabel',
            '-m',
            bool,
            False,
            help='If True, use binary crossentropy instead of categorical',
        ),
    ]

    def run(self, scope, storage):
        max_types = scope['max_types']
        x_scale = self.arguments['x_scale']
        remap = Remap()
        remap('None')

        nlist_generator = scope['nlist_generator']
        frames = []
        for frame in scope['loaded_frames']:
            frame = process_frame(frame, nlist_generator, max_types)
            frames.append(frame)

        if 'pad_size' in scope:
            pad_size = scope['pad_size']
        else:
            pad_size = max(np.max(frame.nlist.neighbor_counts) for frame in frames)

        rng = np.random.default_rng(self.arguments['seed'])

        rs, ts, ws, ys, ctxs = [], [], [], [], []
        for frame in frames:
            samp = np.arange(len(frame.positions))
            if 'subsample' in self.arguments:
                filt = rng.uniform(size=len(samp))
                filt = filt < self.arguments['subsample']
                samp = samp[filt]
                if not len(samp):
                    continue

            (rijs, tijs, wijs) = index_frame(frame, samp, pad_size, 2 * max_types)

            rs.append(rijs)
            ts.append(tijs)
            ws.append(wijs)
            encoded_type = remap(frozenset(frame.context.items()))
            ys.append(np.full_like(rijs[..., :1], encoded_type, dtype=np.int32))
            ctxs.extend(len(rijs) * [frame.context])

        rs = np.concatenate(rs, axis=0)
        ts = np.concatenate(ts, axis=0)
        ws = np.concatenate(ws, axis=0)
        ys = np.concatenate(ys, axis=0)

        rs /= x_scale

        shuf = np.arange(len(rs))
        if self.arguments['shuffle']:
            rng.shuffle(shuf)

        rs = rs[shuf]
        ts = ts[shuf]
        ws = ws[shuf]
        ys = ys[shuf]
        ctxs = np.array(ctxs, dtype=object)[shuf]

        x = [rs, ts, ws] if scope.get('use_bond_weights', False) else [rs, ts]
        y = ys

        loss = 'sparse_categorical_crossentropy'
        if self.arguments['multilabel']:
            onehot = np.eye(len(remap))
            y = onehot[y[..., 0]]
            loss = 'binary_crossentropy'

        scope['x_train'] = x
        scope['y_train'] = y
        scope['x_scale'] = x_scale
        scope['x_contexts'] = ctxs
        scope['loss'] = loss
        scope['label_remap'] = remap
        scope.setdefault('num_classes', len(remap))
        scope.setdefault('metrics', []).append('accuracy')
        scope['multilabel'] = self.arguments['multilabel']
