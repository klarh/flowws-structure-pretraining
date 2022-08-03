import bisect
import collections

import freud
import numpy as np

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
    ],
)


def process_frame(frame, nlist_generator, max_types):
    box = frame.box
    positions = frame.positions
    types = frame.types
    context = frame.context
    nl = nlist_generator(box, positions)

    index_i = nl.query_point_indices
    index_j = nl.point_indices
    weights = nl.weights.copy()
    weights /= np.repeat(np.add.reduceat(weights, nl.segments), nl.neighbor_counts)

    rijs = positions[index_j] - positions[index_i]
    rijs = freud.box.Box.from_box(box).wrap(rijs)
    tijs = encode_types(types[index_i], types[index_j], None, max_types)
    return Frame(
        box, positions, types, context, index_i, index_j, weights, rijs, tijs, nl
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


def index_frame(frame, indices, max_size, type_dim):
    all_bonds = []
    for i in indices:
        bond_start = bisect.bisect_left(frame.index_i, i)
        bond_end = bisect.bisect_left(frame.index_i, i + 1)
        bonds = slice(bond_start, bond_end)
        all_bonds.append(bonds)

    result = [
        pad([frame.rijs[b] for b in all_bonds], max_size, 3),
        pad([frame.tijs[b] for b in all_bonds], max_size, type_dim),
        pad([frame.weights[b] for b in all_bonds], max_size, None),
    ]
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
                yield self.produce(frame_i, particle)
        else:
            for frame_i in range(len(self.frames)):
                for particle in particle_indices[frame_i]:
                    yield self.produce(frame_i, particle)

    def produce(self, frame_i, particle):
        frame = self.frames[frame_i]
        bond_start = bisect.bisect_left(frame.index_i, particle)
        bond_end = bisect.bisect_left(frame.index_i, particle + 1)
        bonds = slice(bond_start, bond_end)

        rijs = frame.rijs[bonds]
        tijs = frame.tijs[bonds]
        weights = frame.weights[bonds]

        return (rijs, tijs, weights), frame.context
