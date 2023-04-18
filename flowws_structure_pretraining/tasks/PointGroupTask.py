import collections
import functools

import flowws
from flowws import Argument as Arg
import numpy as np

from .internal import encode_types
from .point_groups import filter_discrete, PointGroup


class SubgroupClassMap:
    """Generate a list of point-group symmetries and their subgroups.

    This class generates the group-subgroup relationships within a
    specified set of point groups.

    :param n_max: Maximum symmetry degree to consider
    :param blacklist: List of point groups to block from consideration. Can be individual point group names ('C5', 'D2h') or classes ('polyhedral', 'trivial').

    """

    AXIAL_NAMES = [
        n
        for n in PointGroup._REGISTERED_GROUPS
        if any(n.startswith(prefix) for prefix in ['C', 'S', 'D'])
    ]
    POLYHEDRAL_NAMES = [
        n
        for n in PointGroup._REGISTERED_GROUPS
        if any(n.startswith(prefix) for prefix in ['T', 'O', 'I'])
    ]

    def name_expansion(self, name):
        result = []
        if name == 'axial':
            for name in self.AXIAL_NAMES:
                result.extend(self.name_expansion(name))
        elif name == 'polyedral':
            for name in self.POLYHEDRAL_NAMES:
                result.extend(self.name_expansion(name))
        elif name == 'trivial':
            return ['C1']
        elif name == 'redundant':
            return [
                'C1v',
                'C1h',  #'Cs'
                'D1',  #'C2'
                'S2',  #'Ci'
            ]
        elif name == 'Cn':
            return ['C{}'.format(i) for i in range(1, self.n_max + 1)]
        elif name == 'Sn' or name == 'S2n':
            return ['S{}'.format(2 * i) for i in range(1, self.n_max // 2 + 1)]
        elif name == 'Cnh':
            return ['C{}h'.format(i) for i in range(1, self.n_max + 1)]
        elif name == 'Cnv':
            return ['C{}v'.format(i) for i in range(1, self.n_max + 1)]
        elif name == 'Dn':
            return ['D{}'.format(i) for i in range(1, self.n_max + 1)]
        elif name == 'Dnd':
            return ['D{}d'.format(i) for i in range(1, self.n_max + 1)]
        elif name == 'Dnh':
            return ['D{}h'.format(i) for i in range(1, self.n_max + 1)]
        else:
            result.append(name)
        return result

    @classmethod
    def update_subgroups(cls, subgroups, processed=None, focus=None):
        processed = processed or set()

        pending = list(subgroups) if focus is None else [focus]
        while pending:
            to_process = pending.pop()
            if to_process in processed:
                continue
            processed.add(to_process)
            for child in list(subgroups[to_process]):
                cls.update_subgroups(subgroups, processed, child)
                subgroups[to_process].update(subgroups[child])

    def __init__(self, n_max=6, blacklist=['trivial', 'redundant']):
        self.n_max = n_max
        self.blacklist = blacklist

        self.full_blacklist = set()
        for name in blacklist:
            self.full_blacklist.update(self.name_expansion(name))

        column_names = set()
        for group in PointGroup._REGISTERED_GROUPS:
            for name in [
                n for n in self.name_expansion(group) if n not in self.full_blacklist
            ]:
                column_names.add(name)
        self.column_names = sorted_column_names = list(sorted(column_names))
        self.column_name_map = {name: i for (i, name) in enumerate(sorted_column_names)}

        self.subgroups = subgroups = collections.defaultdict(lambda: set(['C1']))
        for name in sorted_column_names:
            subgroups[name].add(name)

        equiv_groups = [
            ['Ci', 'S2'],
            ['C1h', 'C1v', 'Cs'],
            ['D1', 'C2'],
            ['D1h', 'C2v'],
            ['D1d', 'C2h'],
        ]
        for equiv in equiv_groups:
            for name in equiv:
                subgroups[name].update(equiv)

        subgroups['D2'].add('C2')
        subgroups['D2h'].add('C2')

        for n in range(1, n_max + 1):
            child = 'C{}'.format(n)
            parents = [
                name.format(n) for name in ['C{}h', 'C{}v', 'D{}', 'D{}h', 'D{}d']
            ]
            for parent in parents:
                subgroups[parent].add(child)

            names = ['C{}', 'C{}h', 'C{}v', 'D{}', 'D{}h', 'D{}d']
            for name in names:
                for mult in range(2, n_max + 1):
                    if n * mult > self.n_max:
                        break
                    subgroups[name.format(mult * n)].add(name.format(n))

                    base_name = name[:3]
                    subgroups[name.format(mult * n)].add(base_name.format(n))

            subgroups['C{}h'.format(n)].add('Cs')
            subgroups['C{}v'.format(n)].add('Cs')
            subgroups['D{}h'.format(n)].add('Cs')
            subgroups['D{}h'.format(n)].add('C{}h'.format(n))
            subgroups['D{}h'.format(n)].add('C{}v'.format(n))
            subgroups['D{}h'.format(n)].add('D{}'.format(n))
            subgroups['D{}d'.format(n)].add('S{}'.format(2 * n))
            subgroups['D{}d'.format(n)].add('C{}v'.format(n))
            subgroups['D{}h'.format(2 * n)].add('D{}d'.format(n))
            subgroups['S{}'.format(2 * n)].add('C{}'.format(n))

        # n even
        for n in range(2, n_max + 1, 2):
            for mult in range(1, n_max + 1):
                if n * mult > self.n_max:
                    break
                subgroups['S{}'.format(n * mult)].add('S{}'.format(n))
            subgroups['C{}h'.format(n)].add('S{}'.format(n))

        # n odd
        for n in range(1, n_max + 1, 2):
            subgroups['D{}d'.format(n)].add('D{}'.format(n))

        polyhedral_subgroups = {
            'T': ['C3', 'C2', 'D2'],
            'Td': ['C3', 'S4', 'Cs', 'T', 'D2'],
            'Th': ['S6', 'C2', 'Cs', 'T', 'D2', 'D2h'],
            'O': ['C2', 'C3', 'C4', 'D2', 'T', 'D4'],
            'Oh': ['O', 'Cs', 'O', 'D2', 'D2h', 'C4h', 'C4v', 'D4h', 'D2d', 'Td', 'Th'],
            'I': ['D3', 'D5', 'T', 'D2'],
            'Ih': ['C3', 'C5', 'Th', 'D2', 'I'],
        }
        for (name, subs) in polyhedral_subgroups.items():
            for sub in subs:
                subgroups[name].add(sub)

        self.update_subgroups(subgroups)

        subgroup_rows = []
        for name in sorted_column_names:
            row = np.zeros(len(sorted_column_names), dtype=np.int32)
            for j, subname in enumerate(sorted_column_names):
                row[j] = subname in subgroups[name]
            subgroup_rows.append(row)
        self.subgroup_rows = np.array(subgroup_rows)
        self.subgroup_row_map = dict(zip(sorted_column_names, self.subgroup_rows))


class SubgroupDataset:
    """Generate point clouds for training classifiers on point-group symmetries.

    Each element is generated by randomly generating a small cloud of a few points,
    which is then replicated according to the operations of a randomly-selected
    point-group symmetry. Point clouds that exceed a given size are discarded.

    :param n_max: Maximum size of randomly-generated point cloud to be replicated by a selected symmetry operation
    :param sym_max: Maximum symmetry degree to produce
    :param type_max: Maximum number of types to use for point clouds
    :param max_size: Maximum allowed size of replicated point clouds
    :param batch_size: Number of point clouds to generate in each batch
    :param upsample: If True, randomly fill leftover space (up to max_size) in point clouds with identical replicas of points
    :param encoding_filter: Lengthscale to use in `filter_discrete`
    :param blacklist: List of symmetries or symmetry groups to exclude from consideration
    :param seed: RNG seed for generation
    :param multilabel: If True, learn group-subgroup relations in a binary classification setting; if False, learn a single-class classification task
    :param normalize: If True, normalize points to the surface of a sphere; if False, points are allowed to have arbitrary length
    :param sum_difference_types: If True, use a symmetric one-hot sum-difference encoding for type vectors
    :param lengthscale: Lengthscale for generated point clouds

    """

    def __init__(
        self,
        n_max=4,
        sym_max=6,
        type_max=4,
        max_size=32,
        batch_size=16,
        upsample=False,
        encoding_filter=1e-2,
        blacklist=['trivial', 'redundant'],
        seed=13,
        multilabel=False,
        normalize=False,
        sum_difference_types=False,
        lengthscale=1.0,
    ):
        self.n_max = n_max
        self.sym_max = sym_max
        self.type_max = type_max
        self.encoding_filter = encoding_filter
        self.max_size = max_size
        self.batch_size = batch_size
        self.upsample = upsample
        self.multilabel = multilabel
        self.normalize = normalize
        self.sum_difference_types = sum_difference_types
        self.lengthscale = lengthscale

        self.subgroup_transform_getter = functools.lru_cache(PointGroup.get)

        self.subgroup_map = SubgroupClassMap(type_max, blacklist)

        self.train_generator = self.generate(seed)
        self.validation_generator = self.generate(seed + 1)
        self.test_generator = self.generate(seed + 2)

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        type_encoding = np.eye(self.type_max)
        classes = np.array(self.subgroup_map.column_names)
        y_dim = len(self.subgroup_map.column_names)
        orders = {}

        while True:
            batch_r = np.zeros((self.batch_size, self.max_size, 3))
            batch_v = np.zeros((self.batch_size, self.max_size), dtype=np.int32)
            if self.multilabel:
                batch_y = np.zeros((self.batch_size, y_dim), dtype=np.int32)
            else:
                batch_y = np.zeros(self.batch_size, dtype=np.int32)
            i = 0
            while i < self.batch_size:
                name_choice = rng.choice(classes)
                n = min(self.n_max, self.max_size // orders.get(name_choice, 1))
                n = rng.integers(1, max(n, 1), endpoint=True)
                v = rng.integers(0, self.type_max, n)
                r = rng.normal(size=(n, 3))
                symop = self.subgroup_transform_getter(name_choice)
                r = symop(r)
                v = np.tile(v, len(r) // len(v))
                r, v = filter_discrete(r, v, self.encoding_filter)
                orders[name_choice] = len(r) / n
                if len(r) <= self.max_size:
                    batch_r[i, : len(r)] = r
                    batch_v[i, : len(v)] = v[:, 0]

                    if self.upsample and len(r) != self.max_size:
                        delta = self.max_size - len(r)
                        indices = rng.integers(0, len(r), delta)
                        batch_r[i, len(r) :] = r[indices]
                        batch_v[i, len(r) :] = v[indices, 0]
                    if self.multilabel:
                        batch_y[i] = self.subgroup_map.subgroup_row_map[name_choice]
                    else:
                        batch_y[i] = self.subgroup_map.column_name_map[name_choice]
                    i += 1

            if self.normalize:
                batch_r /= np.clip(
                    np.linalg.norm(batch_r, axis=-1, keepdims=True), 1e-7, np.inf
                )
            batch_r *= self.lengthscale

            if self.sum_difference_types:
                source_types = rng.integers(
                    0, self.type_max, (self.batch_size, self.max_size), dtype=np.int32
                )
                batch_v = encode_types(source_types, batch_v, None, self.type_max)
            else:
                batch_v = type_encoding[batch_v]

            yield (batch_r, batch_v), batch_y


@flowws.add_stage_arguments
class PointGroupTask(flowws.Stage):
    """Flowws module to train on a point-group symmetry classification task."""

    ARGS = [
        Arg('batch_size', '-b', int, 8, help='Batch size to produce'),
        Arg(
            'n_max',
            '-n',
            int,
            4,
            help='Maximum number of independent points to produce',
        ),
        Arg(
            'sym_max', '-i', int, 6, help='Maximum degree of axial symmetry to generate'
        ),
        Arg('type_max', '-t', int, 4, help='Maximum number of types to produce'),
        Arg(
            'multilabel',
            '-m',
            bool,
            True,
            help='If True, perform multilabel classification',
        ),
        Arg('max_size', '-z', int, 32, help='Maximum point cloud size to produce'),
        Arg(
            'upsample',
            '-u',
            bool,
            False,
            help='If True, upsample all point clouds to be max_size large',
        ),
        Arg('seed', '-s', int, 13, help='RNG seed to use'),
        Arg(
            'normalize',
            None,
            bool,
            False,
            help='If True, normalize bonds to surface of a sphere',
        ),
        Arg(
            'sum_diff_types',
            None,
            bool,
            False,
            help='If True, use a one-hot sum-difference encoding for types instead of plain one-hot',
        ),
        Arg(
            'lengthscale',
            '-l',
            float,
            1.0,
            help='Lengthscale for generated point clouds',
        ),
        Arg(
            'filter_scale',
            '-f',
            float,
            1e-2,
            help='Distance to use for deduplicating symmetrized points',
        ),
    ]

    def run(self, scope, storage):
        self.dataset = SubgroupDataset(
            self.arguments['n_max'],
            self.arguments['sym_max'],
            self.arguments['type_max'],
            self.arguments['max_size'],
            self.arguments['batch_size'],
            encoding_filter=self.arguments['filter_scale'],
            multilabel=self.arguments['multilabel'],
            seed=self.arguments['seed'],
            normalize=self.arguments['normalize'],
            upsample=self.arguments['upsample'],
            sum_difference_types=self.arguments['sum_diff_types'],
            lengthscale=self.arguments['lengthscale'],
        )

        for key in ['x_train', 'y_train']:
            scope.pop(key, None)
        scope['train_generator'] = self.dataset.train_generator
        scope['validation_generator'] = self.dataset.validation_generator
        scope['test_generator'] = self.dataset.test_generator

        scope['max_types'] = self.arguments['type_max']
        scope['type_embedding_size'] = (
            2 * self.arguments['type_max']
            if self.arguments['sum_diff_types']
            else self.arguments['type_max']
        )
        scope['num_classes'] = len(self.dataset.subgroup_map.column_names)
        scope['per_cloud'] = True
        scope['multilabel'] = self.arguments['multilabel']
        scope['loss'] = (
            'binary_crossentropy'
            if self.arguments['multilabel']
            else 'sparse_categorical_crossentropy'
        )
        scope['multilabel_softmax'] = True
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope.setdefault('metrics', []).append('accuracy')
