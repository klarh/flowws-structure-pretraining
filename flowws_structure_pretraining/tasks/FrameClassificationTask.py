import flowws
from flowws import Argument as Arg
import numpy as np

import tensorflow as tf

from .internal import index_frame, process_frame
from ..internal import Remap


def multilabel_accuracy(y_true, y_pred):
    prod = 0.5 * (y_true * y_pred + (1 - y_true) * (1 - y_pred))
    return tf.math.reduce_mean(prod, axis=-1)


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
        Arg(
            'multilabel_class_mode',
            None,
            str,
            help='If given, set the class probability mode for multilabel classification',
        ),
        Arg(
            'per_cloud',
            '-p',
            bool,
            False,
            help='If True, classify clouds rather than individual bonds',
        ),
        Arg(
            'zero_class',
            '-z',
            bool,
            True,
            help='If False, don\'t create a null class for future unmatched contexts',
        ),
        Arg(
            'entropy_term',
            None,
            float,
            help='If given, use the value to add an entropy-maximizing scaling term to the loss function',
        ),
        Arg(
            'maxp_term',
            None,
            float,
            help='If given, use the value to add a penalty to the maximum predicted class probability',
        ),
        Arg(
            'divergence_term',
            None,
            float,
            help='If given, use the value to add a penalty to class probabilities in the middle of the distribution',
        ),
        Arg(
            'cluster_term',
            None,
            float,
            help='If given, use the value to add a loss penalty for non-close probabilities',
        ),
    ]

    def run(self, scope, storage):
        max_types = scope['max_types']
        x_scale = self.arguments['x_scale']
        remap = Remap()
        if self.arguments['zero_class']:
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

        if self.arguments['per_cloud']:
            ys = ys[..., 0, :].copy()

        x = [rs, ts, ws] if scope.get('use_bond_weights', False) else [rs, ts]
        y = ys

        loss = 'sparse_categorical_crossentropy'
        num_classes = len(remap)
        if self.arguments.get('multilabel', None):
            mode = self.arguments.get('multilabel_class_mode', None)
            if mode is None:
                onehot = np.eye(len(remap))
            elif mode == 'linear':
                dist = np.linspace(0, 1, len(remap), endpoint=False)
                dist = 1 - np.abs(dist[None, :] - dist[:, None])
                dist /= np.sum(dist, axis=-1, keepdims=True)
                onehot = dist
            elif mode == 'nearby':
                N = len(remap)
                encodings = []
                for bandwidth in range(1, N // 2 + 1):
                    for offset in range(bandwidth):
                        classes = (np.arange(N) + offset) // bandwidth
                        class_onehots = np.eye(classes[-1] + 1)
                        encodings.append(class_onehots[classes])
                onehot = np.concatenate(encodings, axis=-1)
                num_classes = onehot.shape[-1]
            elif mode == 'bit_encoding':
                N = len(remap)
                encodings = []
                cur = N
                classes = np.arange(N)
                while cur > 1:
                    class_onehots = np.eye(cur + 1)
                    encodings.append(class_onehots[classes])
                    classes //= 2
                    cur //= 2
                onehot = np.concatenate(encodings, axis=-1)
                num_classes = onehot.shape[-1]
            else:
                raise NotImplementedError(mode)
            y = onehot[y[..., 0]]
            loss = 'binary_crossentropy'
            scope.setdefault('metrics', []).append(multilabel_accuracy)

        if self.arguments.get('entropy_term', None):
            entropy_scale = self.arguments['entropy_term']
            loss_name = loss
            base_loss = tf.keras.losses.get(loss_name)

            def loss(y_true, y_pred):
                base = base_loss(y_true, y_pred)
                if self.arguments.get('multilabel', None):
                    proba = y_pred / tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)
                else:
                    proba = y_pred
                entropy = -tf.math.reduce_sum(proba * tf.math.log(proba), axis=-1)
                # maximize entropy -> negative sign
                return base - entropy_scale * entropy

        elif self.arguments.get('maxp_term', None):
            maxp_scale = self.arguments['maxp_term']
            loss_name = loss
            base_loss = tf.keras.losses.get(loss_name)

            def loss(y_true, y_pred):
                base = base_loss(y_true, y_pred)
                maxp = tf.math.reduce_max(y_pred, axis=-1)
                return base + maxp_scale * maxp

        elif self.arguments.get('divergence_term', None):
            divergence_scale = self.arguments['divergence_term']
            loss_name = loss
            base_loss = tf.keras.losses.get(loss_name)

            def loss(y_true, y_pred):
                base = base_loss(y_true, y_pred)
                maxval = tf.math.reduce_max(y_pred, axis=-1, keepdims=True)
                p0 = y_pred / maxval
                pmax = (maxval - y_pred) / maxval
                divergence = tf.math.reduce_mean(tf.math.minimum(p0, pmax), axis=-1)
                return base + divergence_scale * divergence

        elif self.arguments.get('cluster_term', None):
            cluster_scale = self.arguments['cluster_term']
            loss_name = loss
            base_loss = tf.keras.losses.get(loss_name)

            def loss(y_true, y_pred):
                base = base_loss(y_true, y_pred)
                delta = y_pred[..., None, :] - y_pred[..., :, None]
                cluster = tf.math.reduce_mean(tf.square(delta), axis=(-1, -2))
                return base + cluster_scale * cluster

        for key in ['x_train', 'y_train', 'train_generator', 'validation_generator']:
            scope.pop(key, None)
        scope['x_train'] = x
        scope['y_train'] = y
        scope['x_scale'] = x_scale
        scope['x_contexts'] = ctxs
        scope['loss'] = loss
        scope['label_remap'] = remap
        if not scope.get('freeze_num_classes', False):
            scope['num_classes'] = num_classes
        scope.setdefault('metrics', []).append('accuracy')
        scope['multilabel'] = self.arguments.get('multilabel', None)
        scope['per_cloud'] = self.arguments['per_cloud']
