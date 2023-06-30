import flowws
from flowws import Argument as Arg
import numpy as np

from .FrameClassificationTask import FrameClassificationTask


@flowws.add_stage_arguments
class FrameRegressionTask(FrameClassificationTask):
    """Generate training data to identify from which frame a sample came, as a continuous quantity"""

    ARGS = FrameClassificationTask.ARGS + [
        Arg(
            'context_label',
            None,
            str,
            help='If given, use this value as the key to extract labels from context dictionaries',
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
        Arg('secondary_labels', None, [str], help='Additional context labels to load'),
        Arg(
            'scale_secondary_labels',
            None,
            bool,
            False,
            help='If True, rescale all secondary label values with fitted normalization',
        ),
    ]

    def run(self, scope, storage):
        super().run(scope, storage)
        secondary_labels = self.arguments.get('secondary_labels', [])

        other_ys = []
        if 'context_label' in self.arguments:
            key = self.arguments['context_label']
            ys = [ctx[key] for ctx in scope['x_contexts']]
            for key in secondary_labels:
                if key == 'force':
                    other_ys.append(scope['force_train'])
                else:
                    other_ys.append(np.array([ctx[key] for ctx in scope['x_contexts']]))
        else:
            class_remap = np.linspace(0, 1, len(scope['label_remap']))
            ys = class_remap[scope['y_train'][..., 0]]

        if self.arguments['normalize_labels']:
            from flowws_keras_geometry.data.internal import ScaledMAE

            if 'normalize_label_mean' in scope:
                mu = scope['normalize_label_mean']
                sigma = scope['normalize_label_std']
            elif 'normalize_labels_context_key' in self.arguments:
                key = self.arguments['normalize_labels_context_key']
                filt = [bool(ctx.get(key, False)) for ctx in scope['x_contexts']]
                filtered_ys = np.array(ys)[filt]
                if not len(filtered_ys):
                    raise ValueError('No values found to normalize by')
                mu, sigma = np.mean(filtered_ys), np.std(filtered_ys)
            else:
                mu, sigma = np.mean(ys), np.std(ys)

            ys = (np.array(ys) - mu) / sigma
            if self.arguments['scale_secondary_labels']:
                for v in other_ys:
                    v /= sigma
            for dset_name in scope.get('dataset_names', ['validation', 'test']):
                if dset_name == 'train':
                    # training data will already get handled
                    continue
                dset_name = '{}_data'.format(dset_name)
                if dset_name in scope:
                    scope[dset_name][-1][:] -= mu
                    scope[dset_name][-1][:] /= sigma
            scope['normalize_label_mean'] = mu
            scope['normalize_label_std'] = sigma

            metrics = scope.setdefault('metrics', [])
            if not any(isinstance(m, ScaledMAE) for m in metrics):
                scaled_mae = ScaledMAE(sigma, name='scaled_mae')
                metrics.append(scaled_mae)

        if other_ys:
            scope['y_train'] = [np.asarray(ys)] + other_ys
        else:
            scope['y_train'] = np.asarray(ys)
        scope['loss'] = 'mse'
        scope['metrics'].remove('accuracy')
        if 'mean_absolute_error' not in scope.setdefault('metrics', []):
            scope['metrics'].append('mean_absolute_error')
