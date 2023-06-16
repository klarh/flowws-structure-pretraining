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
    ]

    def run(self, scope, storage):
        metadata = json.loads(scope['metadata.json'])
        self.arg_specifications['layer'].valid_values = flowws.Range(
            0, metadata['num_attention_layers'], (True, False)
        )
        self.per_molecule = metadata['per_molecule']

        key = 'attention_{}'.format(self.arguments['layer'])
        att = scope[key]
        att = att.reshape(scope['{}_shape'.format(key)])

        if att.ndim == 3:
            att = np.mean(att, axis=0)

        positions = scope['position']
        types = scope['type']

        try:
            diameters = scope['diameter']
        except KeyError:
            diameters = 1.0

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

        colors = np.ones((len(types), 4)) * 0.25
        colors[:, :3] = plato.cmap.cubeellipse_intensity(types.astype(np.float32))

        prims = []

        if not self.per_molecule:
            try:
                center_diam = scope['center_diameter']

                centerprim = draw.Spheres(
                    positions=[-com],
                    colors=[(0.5, 0.5, 0.5, 1)],
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
        edge_colors = np.ones((len(edge_starts), 4)) * 0.5
        edge_colors[:, :3] *= att[ei, ej, None]
        edgeprim = draw.Lines(
            start_points=edge_starts,
            end_points=edge_ends,
            colors=edge_colors,
            widths=0.125,
        )
        prims.append(edgeprim)
        colors[:, :3] *= np.diag(att)[:, None] / np.max(np.diag(att))
        prim.colors = colors

        self.scene = draw.Scene(prims, features=dict(additive_rendering=True))

        scope.setdefault('visuals', []).append(self)

    def draw_plato(self):
        return self.scene
