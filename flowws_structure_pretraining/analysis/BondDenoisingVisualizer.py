import functools

import flowws
from flowws import Argument as Arg
import plato
from plato import draw
import numpy as np

from .internal import GeneratorVisualizer
from ..FileLoader import FileLoader


@flowws.add_stage_arguments
class BondDenoisingVisualizer(flowws.Stage, GeneratorVisualizer):
    """Visualize the results of a bond denoising regressor"""

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
        Arg(
            'width',
            None,
            float,
            0.25,
            valid_values=flowws.Range(0, 2.0, True),
            help='Bond rendering width',
        ),
        Arg(
            'cosine_similarity',
            '-c',
            bool,
            False,
            help='Use cosine similarity rather than euclidean distance',
        ),
    ]

    Frame = FileLoader.Frame

    def run(self, scope, storage):
        self.scope = scope
        (positions, starts, ends, theta) = self.get_predictions(scope['cache_key'])
        theta = self.remap_theta(theta)

        colors = plato.cmap.cubehelix(theta)
        colors[:, :3] *= self.arguments['color_scale']
        prim = draw.Lines(
            start_points=starts,
            end_points=ends,
            widths=np.full(len(theta), self.arguments['width']),
            colors=colors,
        )

        scope.setdefault('plato_primitives', []).append(prim)

    def theta_prediction(self, prediction):
        delta = prediction.ends - prediction.starts
        if self.arguments['cosine_similarity']:
            left, right = delta, prediction.theta
            numerator = np.sum(left * right, axis=-1)
            denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
            theta = numerator / denominator
        else:
            delta -= prediction.theta * self.scope['x_scale']
            theta = np.linalg.norm(delta, axis=-1)
        result = prediction._replace(theta=theta)
        return result
