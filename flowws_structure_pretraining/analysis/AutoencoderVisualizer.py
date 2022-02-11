import flowws
from flowws import Argument as Arg
from plato import draw


@flowws.add_stage_arguments
@flowws.register_module
class AutoencoderVisualizer(flowws.Stage):
    """Visualize predictions for an autoencoder task"""

    ARGS = [
        Arg('frame', '-f', int, 0, help='Training data frame index to render'),
        Arg(
            'true_color',
            '-t',
            [float],
            (0.5, 0.1, 0.1, 0.5),
            help='Color to use for the true positions',
        ),
        Arg(
            'prediction_color',
            '-p',
            [float],
            (0.1, 0.1, 0.5, 0.5),
            help='Color to use for the predicted positions',
        ),
    ]

    def run(self, scope, storage):
        self.arg_specifications['frame'].valid_values = flowws.Range(
            0, len(scope['x_train'][0]), (True, False)
        )

        s = slice(self.arguments['frame'], self.arguments['frame'] + 1)
        pred = scope['model'].predict([x[s] for x in scope['x_train']])
        self.prediction = pred
        self.trueval = scope['y_train'][s]
        scope.setdefault('visuals', []).append(self)

    def draw_plato(self):
        trueprim = draw.Spheres(
            positions=self.trueval[0], colors=self.arguments['true_color']
        )
        predprim = draw.Spheres(
            positions=self.prediction[0], colors=self.arguments['prediction_color']
        )
        scene = draw.Scene(
            [trueprim, predprim], zoom=15, features=dict(translucency=True)
        )
        return scene
