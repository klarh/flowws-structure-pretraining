import bisect
import collections
import functools

import flowws
from flowws import Argument as Arg
import freud
import numpy as np

from flowws_keras_geometry.data.internal import DataMixingPool

Frame = collections.namedtuple(
    "Frame",
    [
        "box",
        "positions",
        "types",
        "context",
        "index_i",
        "index_j",
        "weights",
        "rijs",
        "tijs",
        "nlist",
        "forces",
    ],
)


def process_frame(frame, nlist_generator, max_types):
    box = frame.box
    positions = frame.positions
    forces = frame.forces
    types = frame.types
    context = frame.context
    nl = nlist_generator(box, positions, types)

    index_i = nl.query_point_indices
    index_j = nl.point_indices
    weights = nl.weights.copy()
    if len(weights):
        weights /= np.repeat(np.add.reduceat(weights, nl.segments), nl.neighbor_counts)

    rijs = positions[index_j] - positions[index_i]
    rijs = freud.box.Box.from_box(box).wrap(rijs)
    tijs = encode_types(types[index_i], types[index_j], None, max_types)
    return Frame(
        box,
        positions,
        types,
        context,
        index_i,
        index_j,
        weights,
        rijs,
        tijs,
        nl,
        forces,
    )


def encode_types(source_types, dest_types, N, max_types):
    onehot_src = np.eye(max_types)[source_types]
    onehot_dest = np.eye(max_types)[dest_types]

    minus = onehot_dest - onehot_src
    plus = onehot_dest + onehot_src
    if N:
        minus = minus.reshape((-1, N, max_types))
        plus = plus.reshape((-1, N, max_types))

    return np.concatenate([minus, plus], axis=-1)


def pad(xs, max_size, dim=None):
    result = []
    for x in xs:
        if len(x) < max_size:
            padding = np.zeros((max_size - len(x), dim or 1), dtype=x.dtype)
            if dim is None:
                padding = padding[..., 0]
            x = np.concatenate([x, padding], axis=0)
        result.append(x)
    return np.asarray(result)


def index_frame(frame, indices, max_size, type_dim, include_forces=False):
    all_bonds = []
    for i in indices:
        bond_start = bisect.bisect_left(frame.index_i, i)
        bond_end = bisect.bisect_left(frame.index_i, i + 1)
        if bond_start == bond_end:
            continue
        bonds = slice(bond_start, bond_end)
        all_bonds.append(bonds)

    result = [
        pad([frame.rijs[b] for b in all_bonds], max_size, 3),
        pad([frame.tijs[b] for b in all_bonds], max_size, type_dim),
        pad([frame.weights[b] for b in all_bonds], max_size, None),
    ]
    if include_forces:
        forces = frame.forces[indices] if frame.forces is not None else None
        result.append(forces)
    return tuple(result)


class EnvironmentGenerator:
    def __init__(self, frames):
        self.frames = list(frames)
        self.frame_sizes = [len(frame.positions) for frame in self.frames]
        self.frame_probas = np.array(self.frame_sizes, dtype=np.float32) / sum(
            self.frame_sizes
        )

    def sample(self, seed=13, loop=True, subsample=None):
        rng = np.random.default_rng(seed)
        particle_indices = []

        if subsample is not None:
            if np.array(subsample).size == 1:
                if subsample < 0:
                    subsample = (1.0 + subsample, 1)
                else:
                    subsample = (0, subsample)
            left, right = subsample
            for frame in self.frames:
                filt = rng.uniform(size=len(frame.positions))
                filt = np.logical_and(filt >= left, filt < right)
                particle_indices.append(np.where(filt)[0])
        else:
            for frame in self.frames:
                particle_indices.append(np.arange(len(frame.positions)))

        if loop:
            frame_indices = np.arange(len(self.frames))

            while True:
                frame_i = rng.choice(frame_indices, p=self.frame_probas)
                if not len(particle_indices[frame_i]):
                    continue
                particle = rng.choice(particle_indices[frame_i])
                result = self.produce(frame_i, particle)
                if result is not None:
                    yield result
        else:
            for frame_i in range(len(self.frames)):
                for particle in particle_indices[frame_i]:
                    result = self.produce(frame_i, particle)
                    if result is not None:
                        yield result

    def produce(self, frame_i, particle):
        frame = self.frames[frame_i]
        bond_start = bisect.bisect_left(frame.index_i, particle)
        bond_end = bisect.bisect_left(frame.index_i, particle + 1)
        if bond_start == bond_end:
            return None
        bonds = slice(bond_start, bond_end)

        rijs = frame.rijs[bonds]
        tijs = frame.tijs[bonds]
        weights = frame.weights[bonds]

        context = frame.context
        if frame.forces is not None:
            context = dict(frame.context)
            context['force'] = frame.forces[particle]

        return (rijs, tijs, weights), context


class TaskTransformer(flowws.Stage):
    ARGS = [
        Arg('batch_size', '-b', int, 16, help='Number of point clouds in each batch'),
        Arg(
            'pool_size', '-p', int, 512, help='Number of batches to mix in memory pools'
        ),
        Arg(
            'whole_frame',
            None,
            bool,
            False,
            help='If True, use entire frame\'s data as a labeled sample',
        ),
    ]

    def run(self, scope, storage):
        max_types = scope['max_types']
        self.loaded_frames = scope['loaded_frames']
        nlist_generator = scope.get('nlist_generator', None)
        self.use_weights = scope.get('use_bond_weights', False)
        self.frame_modifiers = scope.get('frame_modifiers', [])

        for i, name in enumerate(['train', 'validation', 'test']):
            frame_indices = scope.get('{}_frames'.format(name), None)
            if not frame_indices:
                continue
            target_name = '{}_generator'.format(name)
            evaluate = name == 'test'
            seed = self.arguments['seed'] + i
            data_generator = self.load(
                frame_indices, nlist_generator, max_types, evaluate, seed
            )
            seed ^= 0b101010101
            new_generator = self.transform(data_generator, evaluate, seed)

            data_pool = DataMixingPool(
                self.arguments['pool_size'], self.arguments['batch_size']
            )
            seed *= 2
            pooled_generator = data_pool.sample(new_generator, seed)
            scope[target_name] = map(self.collate_batch, pooled_generator)

        scope['per_molecule'] = self.arguments['whole_frame']

    def collate_batch(self, batch):
        rijs, tijs, wijs, ys = [], [], [], []
        max_size = 0
        if self.use_weights:
            for ((rij, tij, wij), y) in batch:
                rijs.append(rij)
                tijs.append(tij)
                wijs.append(wij)
                ys.append(y)
                max_size = max(max_size, rij.shape[0])

            x = (
                pad(rijs, max_size, 3),
                pad(tijs, max_size, tij.shape[-1]),
                pad(wijs, max_size, None),
            )
        else:
            for ((rij, tij), y) in batch:
                rijs.append(rij)
                tijs.append(tij)
                ys.append(y)
                max_size = max(max_size, rij.shape[0])

            x = (
                pad(rijs, max_size, 3),
                pad(tijs, max_size, tij.shape[-1]),
            )

        if self.per_bond_label:
            y = pad(ys, max_size, ys[0].shape[-1] if ys[0].ndim > 1 else None)
        else:
            if isinstance(ys[0], (list, tuple)):
                y = []
                for i in range(len(ys[0])):
                    values = [v[i] for v in ys]
                    if ys[0][i].ndim > 1:
                        y.append(pad(values, max_size, values[0].shape[-1]))
                    else:
                        y.append(np.array(values))
            else:
                y = np.array(ys)

        return x, y

    def load(self, indices, nlist_generator, max_types, evaluate, seed):
        whole_frame = self.arguments['whole_frame']
        rng = np.random.default_rng(seed)
        frames = np.array(indices, dtype=object)
        done = False

        while not done:
            rng.shuffle(frames)
            for description in frames:
                frame = self.loaded_frames[description.index]
                if not whole_frame:
                    frame = process_frame(frame, nlist_generator, max_types)
                for mod in self.frame_modifiers:
                    frame = mod(frame)

                if whole_frame:
                    rs = frame.positions
                    ts = np.eye(max_types)[frame.types]
                    weights = None
                    context = dict(frame.context)
                    context['force'] = frame.forces
                    yield (rs, ts, weights), context
                else:
                    for i in range(len(frame.positions)):
                        bond_start = bisect.bisect_left(frame.index_i, i)
                        bond_end = bisect.bisect_left(frame.index_i, i + 1)
                        if bond_start == bond_end:
                            continue
                        bonds = slice(bond_start, bond_end)

                        rijs = frame.rijs[bonds]
                        tijs = frame.tijs[bonds]
                        weights = frame.weights[bonds]

                        context = frame.context
                        if frame.forces is not None:
                            context = dict(frame.context)
                            context['force'] = frame.forces[i]

                        yield (rijs, tijs, weights), context

            if evaluate:
                done = True
