import collections

import flowws
from flowws import Argument as Arg
import matplotlib
import numpy as np
import plato

from ..internal import Remap


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
        Arg('component_x', '-x', int, 0, help='Embedding component to plot on x axis'),
        Arg('component_y', '-y', int, 1, help='Embedding component to plot on y axis'),
    ]

    def run(self, scope, storage):
        if 'key' not in self.arguments:
            valid_keys = [
                k
                for (k, v) in sorted(scope.items())
                if k.startswith('embed')
                and isinstance(v, np.ndarray)
                and v.dtype != object
            ]
            key = valid_keys[-1]
        else:
            key = self.arguments['key']

        x = scope[key]

        self.arg_specifications['component_x'].valid_values = flowws.Range(
            0, x.shape[-1], (True, False)
        )
        self.arg_specifications['component_y'].valid_values = flowws.Range(
            0, x.shape[-1], (True, False)
        )

        found_key_values = collections.defaultdict(set)
        embedding_contexts = [dict(d) for d in scope['embedding_contexts']]
        for d in embedding_contexts:
            for (k, v) in d.items():
                found_key_values[k].add(v)
        to_remove = [k for (k, vs) in found_key_values.items() if len(vs) == 1]
        for d in embedding_contexts:
            for k in to_remove:
                d.pop(k, None)

        remap = Remap()
        contexts = np.array([remap(frozenset(d.items())) for d in embedding_contexts])

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

    def get_colormap(self, remap):
        remap_inverse_dicts = [dict(v) for v in remap.inverse]
        file_frames = collections.defaultdict(set)
        get_key = lambda d: d.get('fname', d.get('structure', 'none'))
        get_index = lambda d: d.get('frame', d.get('noise', -1))
        for d in remap_inverse_dicts:
            file_frames[get_key(d)].add(get_index(d))
        file_frames = {k: list(sorted(v)) for (k, v) in file_frames.items()}

        if any(len(v) > 1 for v in file_frames.values()):
            # use special file-frame colormap
            colors = []
            file_starts = dict(
                zip(
                    sorted(file_frames),
                    np.linspace(0, 3, len(file_frames), endpoint=False),
                )
            )
            file_thetas = {
                k: [0.5]
                if len(v) == 1
                else np.linspace(0.2, 0.8, len(v), endpoint=True)
                for (k, v) in file_frames.items()
            }
            file_colors = {
                k: plato.cmap.cubehelix(file_thetas[k], s=file_starts[k], r=0, h=1.2)
                for k in file_frames
            }
            for d in remap_inverse_dicts:
                key = get_key(d)
                index = file_frames[key].index(get_index(d))
                colors.append(file_colors[key][index])
            cmap = matplotlib.colors.ListedColormap(colors)
        elif len(remap) > self.arguments['progressive_threshold']:
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
        return cmap

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()
        remap = self.remap

        cmap = self.get_colormap(remap)
        points = ax.scatter(
            self.x[:, self.arguments['component_x']],
            self.x[:, self.arguments['component_y']],
            c=self.contexts,
            cmap=cmap,
            alpha=0.5,
            vmin=-0.5,
            vmax=len(self.remap) - 0.5,
        )
        if len(remap) > self.arguments['progressive_threshold']:
            cbar = fig.colorbar(points)
        else:
            cbar = fig.colorbar(
                points, ticks=np.linspace(0, len(remap), len(remap), endpoint=False)
            )
            cbar.ax.set_yticklabels(list(map(dict, remap.inverse)))
        cbar.solids.set(alpha=1)
