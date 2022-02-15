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
        model = scope['embedding_model']

        if 'data_generator' in scope:
            self.evaluate_generator(model, scope)
        else:
            self.evaluate(model, scope)

        scope['neighbor_segments'] = np.cumsum(np.insert(scope['neighbor_counts'], 0, 0))[:-1]

        if self.arguments['average_bonds']:
            x = scope['embedding']
            s = scope['neighbor_segments']
            N = scope['neighbor_counts'][:, None]
            scope['embedding'] = np.add.reduceat(x, s)/N
        else:
            scope['embedding_contexts'] = np.repeat(scope['embedding_contexts'], scope['neighbor_counts'])

    def evaluate_generator(self, model, scope):
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
        if xs:
            scope['embedding'] = np.concatenate(xs, axis=0)
            scope['neighbor_counts'] = np.concatenate(counts)
            scope['embedding_contexts'] = ctxs

    def evaluate(self, model, scope):
        pred = model.predict(scope['x_train'], batch_size=self.arguments['batch_size'])
        filt = np.any(scope['x_train'][0] != 0, axis=-1)
        neighbor_count = np.sum(filt, axis=-1)
        scope['embedding'] = pred[filt]
        scope['neighbor_counts'] = neighbor_count
        scope['embedding_contexts'] = scope['x_contexts']
