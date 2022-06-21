import collections

import flowws
from flowws import Argument as Arg
import matplotlib
import numpy as np
import plato

from ..internal import Remap


@flowws.add_stage_arguments
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
        Arg('reverse', '-r', bool, False, help='If True, reverse the colormap'),
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
        unique_contexts = set(map(lambda x: frozenset(x.items()), embedding_contexts))
        context_sort = lambda v: tuple(sorted(v))
        for v in sorted(unique_contexts, key=context_sort):
            remap(v)
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
        left, right = (0.8, 0.2) if self.arguments['reverse'] else (0.2, 0.8)

        if any(len(v) > 1 for v in file_frames.values()):
            # use special file-frame colormap
            colors = []
            ticks = []
            labels = []
            file_starts = dict(
                zip(
                    sorted(file_frames),
                    np.linspace(0, 3, len(file_frames), endpoint=False),
                )
            )
            file_thetas = {
                k: [0.5]
                if len(v) == 1
                else np.linspace(left, right, len(v), endpoint=True)
                for (k, v) in file_frames.items()
            }
            file_colors = {
                k: plato.cmap.cubehelix(file_thetas[k], s=file_starts[k], r=0, h=1.2)
                for k in file_frames
            }
            for d in remap_inverse_dicts:
                key = get_key(d)
                index = file_frames[key].index(get_index(d))
                if index == len(file_frames[key]) // 2:
                    ticks.append(len(colors))
                    labels.append(dict(sorted(d.items())))
                colors.append(file_colors[key][index])
            cmap = matplotlib.colors.ListedColormap(colors)

            if len(ticks) > self.arguments['progressive_threshold']:
                ticks = labels = []

        elif len(remap) > self.arguments['progressive_threshold']:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'custom_cubehelix',
                plato.cmap.cubehelix(
                    np.linspace(left, right, len(remap), endpoint=True)
                ),
            )
            ticks = labels = []
        else:
            cmap = matplotlib.colors.ListedColormap(
                plato.cmap.cubeellipse_intensity(
                    np.linspace(0, 2 * np.pi, len(remap), endpoint=False)
                )
            )
            ticks = np.linspace(0, len(remap), len(remap), endpoint=False)
            labels = [dict(sorted(v)) for v in remap.inverse]
        return cmap, ticks, labels

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()
        remap = self.remap

        cmap, ticks, ticklabels = self.get_colormap(remap)
        points = ax.scatter(
            self.x[:, self.arguments['component_x']],
            self.x[:, self.arguments['component_y']],
            c=self.contexts,
            cmap=cmap,
            alpha=0.5,
            vmin=-0.5,
            vmax=len(self.remap) - 0.5,
        )
        cbar = fig.colorbar(points, ticks=ticks)
        cbar.ax.set_yticklabels(ticklabels)
        cbar.solids.set(alpha=1)
