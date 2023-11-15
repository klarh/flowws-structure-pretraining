import functools

import flowws
from flowws import Argument as Arg
import numpy as np
from plato import draw


@flowws.add_stage_arguments
class AutoencoderVisualizer(flowws.Stage):
    """Visualize predictions for an autoencoder task"""

    ARGS = [
        Arg('frame', '-f', int, 0, help='Training data frame index to render'),
        Arg(
            'true_color',
            '-t',
            (float, float, float, float),
            help='Color to use for the true positions',
        ),
        Arg(
            'prediction_color',
            '-p',
            (float, float, float, float),
            help='Color to use for the predicted positions',
        ),
        Arg('dataset_name', '-d', str, 'data', help='Dataset name to use'),
    ]

    def run(self, scope, storage):
        self.scope = scope

        dataset_name = self.arguments['dataset_name']
        self.static_name = 'x_{}'.format(dataset_name)
        self.static_label_name = 'y_{}'.format(dataset_name)
        self.generator_name = '{}_generator'.format(dataset_name)

        try:
            x, y = self.data_source
        except ValueError:
            x, y, _ = self.data_source

        self.arg_specifications['frame'].valid_values = flowws.Range(
            0, len(x[0]), (True, False)
        )
        true_color = self.arguments.get('true_color', None) or (0.5, 0.1, 0.1, 0.5)
        prediction_color = self.arguments.get('prediction_color', None) or (
            0.1,
            0.1,
            0.5,
            0.5,
        )

        s = slice(self.arguments['frame'], self.arguments['frame'] + 1)
        pred = scope['model'].predict([v[s] for v in x])

        prediction = pred[0].squeeze() * scope['x_scale']
        trueval = y[s][0].squeeze() * scope['x_scale']

        colors = np.repeat([prediction_color, true_color], len(trueval), axis=0)
        scope['color'] = colors
        scope['position'] = np.concatenate([prediction, trueval], axis=0)

        if self.generator_name in scope:
            self.gui_actions = [
                ('Next batch', self._next_batch),
            ]

    @functools.cached_property
    def data_source(self):
        if self.static_name in self.scope:
            return self.scope[self.static_name], self.scope[self.static_label_name]
        return next(self.scope[self.generator_name])

    def _next_batch(self, scope, storage):
        self.data_source = next(self.scope[self.generator_name])
        if scope.get('rerun_callback', None) is not None:
            scope['rerun_callback']()
