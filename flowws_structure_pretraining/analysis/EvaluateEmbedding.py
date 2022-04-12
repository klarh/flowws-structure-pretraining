import functools

import flowws
from flowws import Argument as Arg
import numpy as np


@flowws.add_stage_arguments
class EvaluateEmbedding(flowws.Stage):
    """Evaluate the embedding model on all input data"""

    ARGS = [
        Arg('batch_size', '-b', int, 32, help='Batch size of calculations'),
        Arg(
            'average_bonds',
            '-a',
            bool,
            False,
            help='If True, average per-bond embeddings to be per-particle',
        ),
    ]

    def run(self, scope, storage):
        self.scope = scope
        scope.update(self.value_dict)

        if scope.get('per_cloud', False):
            pass
        elif self.arguments['average_bonds']:
            x = scope['embedding']
            s = scope['neighbor_segments']
            N = np.clip(scope['neighbor_counts'][:, None], 1, 1e8)
            scope['embedding'] = np.add.reduceat(x, s) / N
        else:
            scope['embedding_contexts'] = np.repeat(
                scope['embedding_contexts'], scope['neighbor_counts']
            )

    @functools.cached_property
    def value_dict(self):
        model = self.scope['embedding_model']
        if 'data_generator' in self.scope:
            return self.evaluate_generator(model, self.scope)
        else:
            return self.evaluate(model, self.scope)

    def evaluate_generator(self, model, scope):
        result = {}
        xs = []
        counts = []
        ctxs = []
        neighbor_count = 1
        for (x, y, ctx) in scope['data_generator']:
            pred = model.predict_on_batch(x)
            ndim = pred.shape[-1]
            if not scope.get('per_cloud', False):
                filt = np.any(x[0] != 0, axis=-1)
                neighbor_count = np.sum(filt, axis=-1)
                pred = pred[filt]
            xs.append(pred)
            counts.append(neighbor_count)
            ctxs.extend(ctx)

        result['embedding'] = np.concatenate(xs, axis=0)
        result['embedding_contexts'] = ctxs
        if not scope.get('per_cloud', False):
            result['neighbor_counts'] = np.concatenate(counts)
            result['neighbor_segments'] = np.cumsum(
                np.insert(result['neighbor_counts'], 0, 0)
            )[:-1]
        return result

    def evaluate(self, model, scope):
        result = {}
        pred = model.predict(scope['x_train'], batch_size=self.arguments['batch_size'])
        if not scope.get('per_cloud', False):
            filt = np.any(scope['x_train'][0] != 0, axis=-1)
            pred = pred[filt]
            neighbor_count = np.sum(filt, axis=-1)
            result['neighbor_counts'] = neighbor_count
            result['neighbor_segments'] = np.cumsum(
                np.insert(result['neighbor_counts'], 0, 0)
            )[:-1]
        result['embedding'] = pred
        result['embedding_contexts'] = scope['x_contexts']
        return result
