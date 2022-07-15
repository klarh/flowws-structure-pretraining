import tempfile

import flowws
from flowws import Argument as Arg
import numpy as np

from ..internal import Remap
from .EvaluateEmbedding import EvaluateEmbedding


def cosine_distance(a, b):
    dot = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1)
    denominator = denominator * np.linalg.norm(b, axis=-1)
    return 1 - dot / np.clip(denominator, 1e-7, np.inf)


class AnnoyQuery:
    def __init__(self, refs, metric='euclidean', num_trees=10):
        import annoy

        self.refs = np.asarray(refs)
        self.D = self.refs.shape[-1]
        self.metric = metric
        self.num_trees = num_trees

        self.ann = annoy.AnnoyIndex(self.D, metric=self.metric)
        for (i, v) in enumerate(self.refs):
            self.ann.add_item(i, v)

        self.ann.build(self.num_trees)

        fn = tempfile.NamedTemporaryFile(suffix='_annoy.ann')
        self.ann.save(fn.name)
        self.ann = annoy.AnnoyIndex(self.D, metric=self.metric)
        self.ann.load(fn.name)

    def get_indices(self, vecs, N=None):
        vecs = np.asarray(vecs)
        N = N or vecs.shape[-1]
        indices = [self.ann.get_nns_by_vector(v, N) for v in vecs]
        return np.array(indices, dtype=np.int32)

    def get_coords(self, vecs, N=None):
        vecs = np.asarray(vecs)
        indices = self.get_indices(vecs, N)
        return self.refs[indices]


class PyNNDescentQuery:
    def __init__(self, refs, metric='euclidean'):
        import pynndescent

        self.refs = np.asarray(refs)
        self.D = self.refs.shape[-1]
        self.metric = metric

        self.index = pynndescent.NNDescent(self.refs, self.metric)

    def get_indices(self, vecs, N=None):
        vecs = np.asarray(vecs)
        N = N or vecs.shape[-1]
        (indices, distances) = self.index.query(vecs)
        return indices

    def get_coords(self, vecs, N=None):
        vecs = np.asarray(vecs)
        indices = self.get_indices(vecs, N)
        return self.refs[indices]


@flowws.add_stage_arguments
class EmbeddingDistance(flowws.Stage):
    """Compute statistics on distances in embedding space"""

    ARGS = list(EvaluateEmbedding.ARGS) + [
        Arg('mode', '-m', str, 'nearest', help='Distance calculation mode'),
        Arg('num_trees', None, int, 10, help='Number of annoy trees to generate'),
        Arg('seed', '-s', int, 13, help='RNG seed, if needed'),
        Arg('summarize', None, bool, True, help='Print statistics of number of points'),
        Arg('use_annoy', None, bool, False, help='Use annoy instead of pynndescent'),
    ]

    def run(self, scope, storage):
        mode = self.arguments['mode']
        if self.arguments['use_annoy']:
            metric = 'angular' if mode.endswith('cosine') else 'euclidean'
        else:
            metric = 'cosine' if mode.endswith('cosine') else 'euclidean'
        reference_embedding = scope.get('reference_embedding', scope['embedding'])
        scope['reference_embedding'] = reference_embedding

        child_arg_names = {arg.name for arg in EvaluateEmbedding.ARGS}
        child_args = {k: v for (k, v) in self.arguments.items() if k in child_arg_names}
        EvaluateEmbedding(**child_args).run(scope, storage)
        contexts = scope['embedding_contexts']
        query_embedding = scope['embedding']

        if self.arguments['summarize']:
            print('Reference embedding shape:', reference_embedding.shape)
            print('Query embedding shape:', query_embedding.shape)
            dim = reference_embedding.shape[-1]
            ref_covering = len(reference_embedding) ** (1.0 / dim)
            query_covering = len(query_embedding) ** (1.0 / dim)
            print(
                'Dense embedding dimension covering factors: {} (reference), {}(query)'.format(
                    ref_covering, query_covering
                )
            )

        if 'embedding_distance_query' not in scope:
            if self.arguments['use_annoy']:
                query = AnnoyQuery(
                    reference_embedding, metric, self.arguments['num_trees']
                )
            else:
                query = PyNNDescentQuery(reference_embedding, metric)
            scope['embedding_distance_query'] = query
        else:
            query = scope['embedding_distance_query']

        left = query_embedding
        if mode == 'log_nearest':
            right = query.get_coords(left, 1)[..., 0, :]
            distances = np.linalg.norm(left - right, axis=-1)
            distances = np.log(distances)
        elif mode == 'avg_log_simplex':
            right = query.get_coords(left, left.shape[-1])
            left = query_embedding[..., None, :]
            delta = np.linalg.norm(right - left, axis=-1)
            distances = np.nanmean(np.log(delta), axis=-1)
        elif mode.startswith('nearest'):
            right = query.get_coords(left, 1)[..., 0, :]

            if mode == 'nearest':
                distances = np.linalg.norm(left - right, axis=-1)
            elif mode == 'nearest_cosine':
                distances = cosine_distance(left, right)
            else:
                raise NotImplementedError(mode)
        elif mode.startswith('avg_simplex'):
            right = query.get_coords(left, left.shape[-1])
            left = query_embedding[..., None, :]
            if mode == 'avg_simplex':
                delta = np.linalg.norm(right - left, axis=-1)
                distances = np.mean(delta, axis=-1)
            elif mode == 'avg_simplex_cosine':
                distances = cosine_distance(left, right)
                distances = np.mean(distances, axis=-1)
            else:
                raise NotImplementedError(mode)
        elif mode.endswith('det_simplex'):
            right = query.get_coords(left, left.shape[-1])
            left = query_embedding[..., None, :]
            delta = right - left

            if mode == 'det_simplex':
                distances = np.abs(np.linalg.det(delta))
            elif mode == 'log_det_simplex':
                distances = np.linalg.slogdet(delta)[1]
            else:
                raise NotImplementedError(mode)
        else:
            raise NotImplementedError(mode)

        remap = Remap()
        unique_contexts = set(map(lambda x: frozenset(x.items()), contexts))
        context_sort = lambda v: tuple(sorted(v))
        for v in sorted(unique_contexts, key=context_sort):
            remap(v)
        contexts = np.array([remap(frozenset(d.items())) for d in contexts])

        sort = np.argsort(contexts)
        sorted_distances = distances[sort]
        sorted_contexts = contexts[sort]
        splits = np.unique(sorted_contexts, return_index=True)[1][1:]
        context_observations = np.split(sorted_distances, splits)
        self.remap = remap
        self.means = [np.nanmean(vs) for vs in context_observations]

        scope['reference_distances'] = distances
        scope['embedding_distance_remap'] = self.remap
        scope['embedding_distance_context_labels'] = self.remap.inverse
        scope['embedding_distance_contexts'] = contexts
        scope['embedding_distance_means'] = self.means
        scope.setdefault('visuals', []).append(self)

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()
        ax.plot(self.means)
        ax.set_xlabel('Context index')
        ax.set_ylabel('Mean distance')
