import functools

import flowws
from flowws import Argument as Arg
import plato
from plato import draw
import numpy as np

from .internal import GeneratorVisualizer
from ..FileLoader import FileLoader


@flowws.add_stage_arguments
class ShiftIdentificationVisualizer(flowws.Stage, GeneratorVisualizer):
    """Visualize the results of a shift identification regressor"""

    ARGS = [
        Arg(
            'color_scale',
            None,
            float,
            1,
            valid_values=flowws.Range(0, 10, True),
            help='Factor to scale color RGB intensities by',
        ),
        Arg(
            'reverse',
            '-r',
            bool,
            False,
            help='If True, reverse classification colormap',
        ),
        Arg('mode', '-m', str, help='Colormap mode'),
        Arg('color_min', None, float, 0.0, help='Minimum colormap value'),
        Arg('color_max', None, float, 1.0, help='Maximum colormap value'),
        Arg('contrast', None, float, 1.0, help='Contrast scale'),
    ]

    Frame = FileLoader.Frame

    def run(self, scope, storage):
        self.scope = scope
        (positions, starts, ends, theta) = self.get_predictions(scope['cache_key'])
        theta = self.remap_theta(theta)

        colors = plato.cmap.cubehelix(theta)
        colors[:, :3] *= self.arguments['color_scale']
        scope['color'] = colors

    @staticmethod
    def theta_prediction(prediction):
        return prediction._replace(theta=np.linalg.norm(prediction.theta, axis=-1))

    @staticmethod
    def filter_prediction(pred, filt):
        return pred
