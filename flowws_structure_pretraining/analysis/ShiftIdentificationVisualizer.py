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
        Arg('draw_shift', None, bool, False, help='Render shifts as lines'),
        Arg(
            'width',
            None,
            float,
            0.25,
            valid_values=flowws.Range(0, 2.0, True),
            help='Bond rendering width',
        ),
    ]

    Frame = FileLoader.Frame

    def run(self, scope, storage):
        self.scope = scope
        (positions, starts, ends, theta) = self.get_predictions(scope['cache_key'])
        if self.arguments['draw_shift']:
            starts = scope['position']
            ends = starts + self.last_prediction
            theta = self.remap_theta(theta)

            colors = plato.cmap.cubehelix(theta)
            colors[:, :3] *= self.arguments['color_scale']
            prim = draw.Lines(
                start_points=starts,
                end_points=ends,
                widths=np.full(len(theta), self.arguments['width']),
                colors=colors,
                cap_mode=1,
            )

            scope.setdefault('plato_primitives', []).append(prim)
        else:
            theta = self.remap_theta(theta)

            colors = plato.cmap.cubehelix(theta)
            colors[:, :3] *= self.arguments['color_scale']
            scope['color'] = colors

    def theta_prediction(self, prediction):
        self.last_prediction = prediction.theta
        return prediction._replace(theta=np.linalg.norm(prediction.theta, axis=-1))

    @staticmethod
    def filter_prediction(pred, filt):
        return pred
