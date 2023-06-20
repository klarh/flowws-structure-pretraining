import contextlib
import os
import json

import numpy as np
from tensorflow import keras

import flowws
from flowws import Argument as Arg
import garnett
import gtar

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x):
        for v in x:
            yield v


@flowws.add_stage_arguments
class ExportAttentionMaps(flowws.Stage):
    """Export the evaluated attention maps of a trained model as a trajectory."""

    ARGS = [
        Arg(
            'layer',
            '-l',
            int,
            help='If given, specify a particular attention layer index to use',
        ),
        Arg(
            'trajectory_filenames',
            None,
            [(str, str)],
            help='List of (filename, directory) pairs to find input trajectory files',
        ),
        Arg(
            'filename',
            '-f',
            str,
            'attention.sqlite',
            help='Filename to dump attention maps to',
        ),
        Arg('batch_size', '-b', int, 1, help='Batch size for attention map evaluation'),
        Arg('dataset', '-d', str, 'train', help='Dataset name to evaluate'),
        Arg(
            'atomic_diameters',
            None,
            bool,
            False,
            help='If True, use atomic type names to export diameters',
        ),
        Arg(
            'sort_contexts',
            None,
            bool,
            False,
            help='If True, sort observations by input context before dumping',
        ),
    ]

    def run(self, scope, storage):
        dset_name = '{}_data'.format(self.arguments['dataset'])
        (dataset, _) = scope[dset_name]

        context_name = (
            'x' if self.arguments['dataset'] == 'train' else self.arguments['dataset']
        )
        contexts = scope['{}_contexts'.format(context_name)]

        attention_layers = scope['attention_outputs']
        if 'layer' in self.arguments:
            attention_layers = [attention_layers[self.arguments['layer']]]

        trajectory_map = dict(self.arguments['trajectory_filenames'])

        inputs = scope['input_symbol']
        model = keras.models.Model(inputs, attention_layers[::-1])
        layer_names = [l.node.outbound_layer.name for l in attention_layers[::-1]]

        self.per_molecule = scope.get('per_molecule', False)
        self.x_scale = scope.get('x_scale', 1.0)

        metadata = dict(
            num_attention_layers=len(attention_layers),
            old_metadata=scope.get('metadata', {}),
            per_molecule=self.per_molecule,
        )

        rev_type_map = {i: t for (t, i) in scope['type_name_map'].items()}
        type_names = [rev_type_map[i] for i in range(len(rev_type_map))]

        self.type_diameters = None
        if self.arguments.get('atomic_diameters', False):
            import mendeleev

            self.type_diameters = (
                2
                * np.array(
                    [
                        mendeleev.element(type_names[i]).atomic_radius
                        for i in range(len(type_names))
                    ]
                )
                / 100
            )

        i = 0
        with contextlib.ExitStack() as stack:
            ref_trajectories = {}
            for (name, path) in trajectory_map.items():
                if os.path.exists(os.path.join(path, name)):
                    ref_trajectories[name] = stack.enter_context(
                        garnett.read(os.path.join(path, name))
                    )
                else:
                    ref_trajectories[name] = stack.enter_context(garnett.read(path))

            traj = stack.enter_context(gtar.GTAR(self.arguments['filename'], 'w'))

            traj.writePath('metadata.json', json.dumps(metadata))
            traj.writePath('type_names.json', json.dumps(type_names))
            traj.writePath('layer_names.json', json.dumps(layer_names))
            for (data, context, pred) in self.batch(model, dataset, contexts):
                i += self.write(traj, data, context, pred, i, ref_trajectories)

    def batch(self, model, dataset, contexts):
        N = len(dataset[0])
        batch_size = self.arguments['batch_size']
        sort_contexts = self.arguments['sort_contexts']

        if sort_contexts:
            ctx_arr = np.empty(len(contexts), dtype=object)
            ctx_arr[:] = [tuple(sorted((c.items()))) for c in contexts]
            sortidx = np.argsort(ctx_arr)

        for i in tqdm(range(0, N, batch_size)):
            sel = slice(i, i + batch_size)

            if sort_contexts:
                idx = sortidx[sel]
                x = [v[idx] for v in dataset]
                ctx = contexts[idx]
            else:
                x = [v[sel] for v in dataset]
                ctx = contexts[sel]
            prediction = model.predict(x, verbose=False, batch_size=batch_size)
            yield x, ctx, prediction

    def write(self, traj, data, context, prediction, offset, ref_trajectories):

        for layer_index, batch_pred in enumerate(prediction):
            base_name = 'attention_{}'.format(layer_index)
            att_name = '{}.f32.uni'.format(base_name)
            shape_name = '{}_shape.u32.uni'.format(base_name)
            for index, (pred, rijs, tijs, ctx) in enumerate(
                zip(batch_pred, *data, context)
            ):
                pred = pred.squeeze()

                filt = np.any(tijs != 0, axis=-1)
                N = np.sum(filt)
                pred = pred[tuple(slice(0, N) for _ in range(pred.ndim))]
                rijs = rijs[filt] * self.x_scale
                if self.per_molecule:
                    tj = np.argmax(tijs, axis=-1)[filt]
                    ti = np.full_like(tj, -1)
                else:
                    minus, plus = (
                        tijs[:, : tijs.shape[1] // 2],
                        tijs[:, tijs.shape[1] // 2 :],
                    )
                    ti = np.argmax(plus - minus, axis=-1)
                    tj = np.argmax(plus + minus, axis=-1)
                    ti, tj = ti[filt], tj[filt]

                traj_fname = os.path.basename(ctx['fname'])
                if traj_fname in ref_trajectories:
                    frame = ref_trajectories[traj_fname][ctx['frame']]
                    box = frame.box

                    boxarr = [box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz]
                    record_name = 'frames/{}/box.f32.uni'.format(offset + index)
                    traj.writePath(record_name, boxarr)

                if self.type_diameters is not None:
                    record_name = 'frames/{}/diameter.f32.ind'.format(offset + index)
                    traj.writePath(record_name, self.type_diameters[tj])
                    if ti[0] != -1:
                        record_name = 'frames/{}/center_diameter.f32.uni'.format(
                            offset + index
                        )
                        traj.writePath(record_name, self.type_diameters[ti[0]])

                record_name = 'frames/{}/context.json'.format(offset + index)
                traj.writePath(record_name, json.dumps(ctx))
                record_name = 'frames/{}/position.f32.ind'.format(offset + index)
                traj.writePath(record_name, rijs)
                record_name = 'frames/{}/type.i32.ind'.format(offset + index)
                traj.writePath(record_name, tj)
                record_name = 'frames/{}/source_type.i32.ind'.format(offset + index)
                traj.writePath(record_name, ti)
                record_name = 'frames/{}/{}'.format(offset + index, shape_name)
                traj.writePath(record_name, pred.shape)
                record_name = 'frames/{}/{}'.format(offset + index, att_name)
                traj.writePath(record_name, pred)
        return len(data[0])
