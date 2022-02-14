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

        if self.arguments['average_bonds']:
            if scope['embedding'].ndim == 3:
                scope['embedding'] = np.mean(scope['embedding'], axis=1)

    def evaluate_generator(self, model, scope):
        xs = []
        ctxs = []
        for (x, y, ctx) in scope['data_generator']:
            xs.append(model.predict_on_batch(x))
            ctxs.extend(ctx)
        if xs:
            xs = np.concatenate(xs, axis=0)
            scope['embedding'] = xs
            scope['embedding_contexts'] = ctxs

    def evaluate(self, model, scope):
        xs = model.predict(scope['x_train'], batch_size=self.arguments['batch_size'])
        scope['embedding'] = xs
        scope['embedding_contexts'] = scope['x_contexts']
