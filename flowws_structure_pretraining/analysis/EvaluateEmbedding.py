import functools

import flowws
from flowws import Argument as Arg
import numpy as np


@flowws.add_stage_arguments
@flowws.register_module
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

        if self.arguments['average_bonds']:
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
        for (x, y, ctx) in scope['data_generator']:
            pred = model.predict_on_batch(x)
            ndim = pred.shape[-1]
            filt = np.any(x[0] != 0, axis=-1)
            neighbor_count = np.sum(filt, axis=-1)
            xs.append(pred[filt])
            counts.append(neighbor_count)
            ctxs.extend(ctx)

        result['embedding'] = np.concatenate(xs, axis=0)
        result['neighbor_counts'] = np.concatenate(counts)
        result['embedding_contexts'] = ctxs
        result['neighbor_segments'] = np.cumsum(
            np.insert(result['neighbor_counts'], 0, 0)
        )[:-1]
        return result

    def evaluate(self, model, scope):
        result = {}
        pred = model.predict(scope['x_train'], batch_size=self.arguments['batch_size'])
        filt = np.any(scope['x_train'][0] != 0, axis=-1)
        neighbor_count = np.sum(filt, axis=-1)
        result['embedding'] = pred[filt]
        result['neighbor_counts'] = neighbor_count
        result['embedding_contexts'] = scope['x_contexts']
        result['neighbor_segments'] = np.cumsum(
            np.insert(result['neighbor_counts'], 0, 0)
        )[:-1]
        return result
