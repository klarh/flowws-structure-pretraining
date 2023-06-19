import json

import freud
import numpy as np
import plato, plato.draw as draw
import flowws
from flowws import Argument as Arg


def percentile_remap(x):
    shape = x.shape
    result = np.empty_like(x, dtype=np.float32)
    sortidx = np.argsort(x.reshape(-1))
    result.flat[sortidx] = np.linspace(0, 1.0, x.size)
    return result


class AttentionMapVisual:
    def __init__(self):
        pass

    def update(self, metadata, scope, selected_layer):
        num_layers = metadata['num_attention_layers']
        self.shapes = [scope['attention_{}_shape'.format(i)] for i in range(num_layers)]
        self.names = json.loads(scope['layer_names.json'])
        self.selected_layer = selected_layer

    def draw_matplotlib(self, figure):
        labels = [
            '{} ({})'.format(name, shape)
            for (name, shape) in zip(self.names, self.shapes)
        ]
        colors = np.tile([(0.8, 0.4, 0.4, 1)], (len(labels), 1))
        colors[self.selected_layer] = (0.4, 0.4, 0.8, 1)
        ys = np.linspace(0, 1, len(labels) + 2, endpoint=True)[1:-1]

        for (y, label, color) in zip(ys, labels, colors):
            figure.text(0.5, y, label, color=color, ha='center', va='center')


@flowws.add_stage_arguments
class ViewAttentionMap(flowws.Stage):
    """Visualize a set of evaluated attention maps."""

    ARGS = [
        Arg(
            'layer',
            '-l',
            int,
            0,
            help='Attention layer index to use (starting with the final layer)',
        ),
        Arg('center', '-c', bool, True, help='If True, center particles'),
        Arg('additive_rendering', None, bool, True, help='Utilize additive rendering'),
        Arg(
            'fast_antialiasing',
            None,
            bool,
            False,
            help='Use Fast Approximate Antialiasing (FXAA)',
        ),
        Arg(
            'color_scale',
            None,
            float,
            1,
            valid_values=flowws.Range(0, 10, True),
            help='Factor to scale color RGB intensities by',
        ),
        Arg('draw_scale', '-s', float, 1, help='Scale to multiply particle size by'),
        Arg('display_box', '-b', bool, True, help='Display the system box'),
        Arg(
            'focus_particle',
            '-f',
            int,
            -1,
            help='Particle to focus on for 3D attention maps',
        ),
    ]

    def run(self, scope, storage):
        metadata = json.loads(scope['metadata.json'])
        self.arg_specifications['layer'].valid_values = flowws.Range(
            0, metadata['num_attention_layers'], (True, False)
        )
        self.per_molecule = metadata['per_molecule']

        if not hasattr(self, 'layer_visual'):
            self.layer_visual = AttentionMapVisual()
        self.layer_visual.update(metadata, scope, self.arguments['layer'])

        key = 'attention_{}'.format(self.arguments['layer'])
        att = scope[key]
        att = att.reshape(scope['{}_shape'.format(key)])

        self.arg_specifications['focus_particle'].valid_values = flowws.Range(
            -1, 32, (True, False)
        )

        focused_on = None
        if att.ndim == 3:
            if self.arguments['focus_particle'] >= 0:
                focused_on = self.arguments['focus_particle'] % len(att)
                att = att[focused_on]
            else:
                att = np.mean(att, axis=0)

        positions = scope['position']
        types = scope['type']

        try:
            diameters = scope['diameter']
        except KeyError:
            diameters = 1.0
        diameters *= self.arguments['draw_scale']

        try:
            fbox = freud.box.Box.from_box(scope['box'])
        except KeyError:
            fbox = None

        if not fbox:
            com = np.zeros(3)
        elif self.per_molecule:
            com = fbox.center_of_mass(positions)
        else:
            com = fbox.center_of_mass(np.concatenate([positions, [(0, 0, 0)]], axis=0))

        if self.arguments['center'] and fbox is not None:
            positions -= com
            positions = fbox.wrap(positions)

        colors = np.ones((len(types), 4))
        colors[:, :3] = plato.cmap.cubeellipse_intensity(types.astype(np.float32))
        colors[:, :3] *= self.arguments['color_scale']

        prims = []
        scene_kwargs = {}

        if not self.per_molecule:
            try:
                center_diam = scope['center_diameter']
                gray = np.mean(colors[:, :3])
                gray = (gray, gray, gray, 1)

                centerprim = draw.Spheres(
                    positions=[-com],
                    colors=[gray],
                    diameters=center_diam,
                )
                prims.append(centerprim)
            except KeyError:
                pass

        prim = draw.Spheres(positions=positions, colors=colors, diameters=diameters)
        prims.append(prim)

        ei, ej = np.where(percentile_remap(att) > 0.5)
        edge_starts = positions[ei]
        edge_ends = positions[ej]
        if fbox is not None:
            edge_ends = edge_starts + fbox.wrap(edge_ends - edge_starts)
        edge_colors = np.ones((len(edge_starts), 4))
        edge_colors[:, :3] *= att[ei, ej, None] * np.mean(colors[:, :3])
        edgeprim = draw.Lines(
            start_points=edge_starts,
            end_points=edge_ends,
            colors=edge_colors,
            widths=0.125,
            cap_mode=1,
        )
        prims.append(edgeprim)
        colors[:, :3] *= np.diag(att)[:, None]
        if focused_on is not None:
            colors[focused_on] = (0.5, 0.5, 0.5, 1)
        prim.colors = colors

        if 'box' in scope and self.arguments['display_box']:
            boxprim = draw.Box.from_box(scope['box'])
            boxprim.width = min(scope['box'][:3]) * 5e-3
            gray = np.mean(colors[:, :3])
            boxprim.color = (gray, gray, gray, 1)
            prims.append(boxprim)

            scene_kwargs['size'] = 1.05 * np.array(scope['box'][:2])
            # use default size of 800px wide
            scene_kwargs['pixel_scale'] = 800 / scene_kwargs['size'][0]

        features = {}

        if self.arguments['additive_rendering']:
            features['additive_rendering'] = True

        if self.arguments['fast_antialiasing']:
            features['fxaa'] = True

        self.scene = draw.Scene(prims, features=features, **scene_kwargs)

        scope.setdefault('visuals', []).append(self)
        scope['visuals'].append(self.layer_visual)

    def draw_plato(self):
        return self.scene
