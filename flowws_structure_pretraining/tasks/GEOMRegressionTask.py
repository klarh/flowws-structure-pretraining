import flowws
import numpy as np
from flowws import Argument as Arg

import tensorflow as tf

from ..internal import Remap
from .FrameClassificationTask import FrameClassificationTask
from .internal import index_frame, process_frame
from ..NearestNeighbors import NearestNeighbors


@flowws.add_stage_arguments
class GEOMRegressionTask(FrameClassificationTask):
    """Generate training data to identify from which frame a sample came, as a continuous quantity"""

    ARGS = [
        Arg(
            'x_scale', '-x', float, 2.0, help='Scale by which to divide input distances'
        ),
        Arg('seed', '-s', int, 13, help='RNG seed for data generation'),
        Arg('subsample', None, float, help='Take only the given fraction of data'),
        Arg('shuffle', None, bool, True, help='If True, shuffle data'),
        Arg(
            'per_cloud',
            '-p',
            bool,
            False,
            help='If True, classify clouds rather than individual bonds',
        ),
    ]

    def run(self, scope, storage):
        """run.

        Parameters
        ----------
        scope :
            scope
        storage :
            storage
        """
        super().run(scope, storage)
        scope['y_train'] = np.array([[item['relativeenergy']] for item in scope['x_contexts']])
        scope['y_train'] = scope['y_train'].repeat(scope['x_train'][0].shape[1], axis=1)

        scope['loss'] = 'mse'
        scope['metrics'].remove('accuracy')
        scope['metrics'].append('mean_absolute_error')
        self.process_val_data(scope, storage)

    def process_val_data(self, scope, storage):
        """process_val_data.

        Parameters
        ----------
        scope :
            scope
        storage :
            storage
        """
        max_types = scope['max_types']
        x_scale = self.arguments['x_scale']
        remap = Remap()
        if 'zero_class' in self.arguments and self.arguments['zero_class']:
            remap('None')

        #nlist_generator = self.val_nlist_generator
        nlist_generator = scope['nlist_generator']
        frames = []
        for frame in scope['validation_data']:
            frame = process_frame(frame, nlist_generator, max_types)
            frames.append(frame)

        if 'pad_size' in scope:
            pad_size = scope['pad_size']
        else:
            pad_size = max(np.max(frame.nlist.neighbor_counts)
                           for frame in frames)

        rng = np.random.default_rng(self.arguments['seed'])

        rs, ts, ws, ctxs = [], [], [], []
        for frame in frames:
            samp = np.arange(len(frame.positions))
            if 'subsample' in self.arguments:
                filt = rng.uniform(size=len(samp))
                filt = filt < self.arguments['subsample']
                samp = samp[filt]
                if not len(samp):
                    continue

            (rijs, tijs, wijs) = index_frame(
                frame, samp, pad_size, 2 * max_types)

            rs.append(rijs)
            ts.append(tijs)
            ws.append(wijs)
            ctxs.extend(len(rijs) * [frame.context])

        rs = np.concatenate(rs, axis=0)
        ts = np.concatenate(ts, axis=0)
        ws = np.concatenate(ws, axis=0)

        rs /= x_scale

        shuf = np.arange(len(rs))
        if self.arguments['shuffle']:
            rng.shuffle(shuf)

        rs = rs[shuf]
        ts = ts[shuf]
        ws = ws[shuf]
        ctxs = np.array(ctxs, dtype=object)[shuf]

        x = [rs, ts, ws] if scope.get('use_bond_weights', False) else [rs, ts]
        y = np.array([[item['relativeenergy']] for item in ctxs])
        y = y.repeat(scope['x_train'][0].shape[1], axis=1)
        print('loading val data')
        scope['validation_data'] = (x,y) 
