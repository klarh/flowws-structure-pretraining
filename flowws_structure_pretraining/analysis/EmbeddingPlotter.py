import collections

import flowws
from flowws import Argument as Arg
import matplotlib
import numpy as np
import plato


class Remap:
    def __init__(self):
        self.m = collections.defaultdict(lambda: len(self.m))

    def __call__(self, x):
        return self.m[x]

    @property
    def inverse(self):
        s = sorted([v, k] for (k, v) in self.m.items())
        return [v[1] for v in s]

    def __len__(self):
        return len(self.m)


@flowws.add_stage_arguments
@flowws.register_module
class EmbeddingPlotter(flowws.Stage):
    """Use PCA to project the embedding"""

    ARGS = [
        Arg('key', '-k', str, help='Key to use for embedding'),
        Arg('shuffle', '-s', bool, True, help='Shuffle points before plotting'),
        Arg('seed', None, int, 13, help='RNG seed for random shuffling'),
        Arg(
            'progressive_threshold',
            '-p',
            int,
            16,
            help='If more keys than this are given, use a progressive colormap',
        ),
    ]

    def run(self, scope, storage):
        if 'key' not in self.arguments:
            valid_keys = [
                k
                for (k, v) in sorted(scope.items())
                if k.startswith('embed') and isinstance(v, np.ndarray)
            ]
            key = valid_keys[-1]
        else:
            key = self.arguments['key']

        x = scope[key]
        remap = Remap()
        contexts = np.array(
            [remap(frozenset(d.items())) for d in scope['embedding_contexts']]
        )

        if x.ndim > 2:
            contexts = np.repeat(contexts, x.shape[-2])
            x = x.reshape((-1, x.shape[-1]))

        if self.arguments['shuffle']:
            rng = np.random.default_rng(self.arguments['seed'])
            shuf = np.arange(len(x))
            rng.shuffle(shuf)
            x = x[shuf]
            contexts = contexts[shuf]

        self.remap = remap
        self.x = x
        self.contexts = contexts

        scope.setdefault('visuals', []).append(self)

    def plot_matplotlib(self, fig):
        ax = fig.add_subplot()
        remap = self.remap

        progressive = len(remap) > self.arguments['progressive_threshold']
        if progressive:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'custom_cubehelix',
                plato.cmap.cubehelix(np.linspace(0.2, 0.8, len(remap), endpoint=True)),
            )
        else:
            cmap = matplotlib.colors.ListedColormap(
                plato.cmap.cubeellipse_intensity(
                    np.linspace(0, 2 * np.pi, len(remap), endpoint=False)
                )
            )
        points = ax.scatter(
            self.x[:, 0],
            self.x[:, 1],
            c=self.contexts,
            cmap=cmap,
            alpha=0.5,
            vmin=-0.5,
            vmax=len(self.remap) - 0.5,
        )
        if progressive:
            cbar = fig.colorbar(points)
        else:
            cbar = fig.colorbar(
                points, ticks=np.linspace(0, len(remap), len(remap), endpoint=False)
            )
            cbar.ax.set_yticklabels(remap.inverse)
