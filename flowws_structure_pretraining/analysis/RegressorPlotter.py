import collections
import functools

import flowws
from flowws import Argument as Arg
import matplotlib
import numpy as np

from ..internal import Remap


@flowws.add_stage_arguments
class RegressorPlotter(flowws.Stage):
    """Plot regression observations for each input frame"""

    ARGS = [
        Arg('batch_size', '-b', int, 32, help='Batch size of calculations'),
    ]

    def run(self, scope, storage):
        self.scope = scope
        scope.update(self.value_dict)

        found_key_values = collections.defaultdict(set)
        regressor_contexts = [dict(d) for d in scope['regressor_contexts']]
        for d in regressor_contexts:
            for (k, v) in d.items():
                found_key_values[k].add(v)
        to_remove = [k for (k, vs) in found_key_values.items() if len(vs) == 1]
        for d in regressor_contexts:
            for k in to_remove:
                d.pop(k, None)

        remap = Remap()
        unique_contexts = set(map(lambda x: frozenset(x.items()), regressor_contexts))
        context_sort = lambda v: tuple(sorted(v))
        for v in sorted(unique_contexts, key=context_sort):
            remap(v)
        contexts = np.array([remap(frozenset(d.items())) for d in regressor_contexts])

        predictions = scope['predictions']
        binned_predictions = collections.defaultdict(list)

        for (ctx, pred) in zip(contexts, predictions):
            binned_predictions[ctx].append(pred)

        self.binned_predictions = binned_predictions
        scope['binned_predictions'] = self.binned_predictions

        scope.setdefault('visuals', []).append(self)

    @functools.cached_property
    def value_dict(self):
        model = self.scope['model']
        if 'data_generator' in self.scope:
            return self.evaluate_generator(model, self.scope)
        else:
            return self.evaluate(model, self.scope)

    def evaluate_generator(self, model, scope):
        result = {}
        xs = []
        ctxs = []
        for (x, y, ctx) in scope['data_generator']:
            pred = model.predict_on_batch(x)
            xs.append(pred[filt])
            ctxs.extend(ctx)

        result['predictions'] = np.concatenate(xs, axis=0)
        result['regressor_contexts'] = ctxs
        return result

    def evaluate(self, model, scope):
        result = {}
        xs = []
        preds = []
        (rs, ts) = scope['x_train']
        for i_start in range(0, len(rs), self.arguments['batch_size']):
            batch = slice(i_start, i_start + self.arguments['batch_size'])
            x = rs[batch], ts[batch]
            pred = model.predict_on_batch(x)
            xs.append(pred)

        result['predictions'] = np.concatenate(xs, axis=0)
        result['regressor_contexts'] = scope['x_contexts']
        return result

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()

        xs, mu, err = [], [], []
        for x in list(sorted(self.binned_predictions)):
            preds = self.binned_predictions[x]
            xs.append(x)
            mu.append(np.mean(preds))
            err.append(np.std(preds) / np.sqrt(len(preds)))
        xs, mu, err = np.array(xs), np.array(mu), np.array(err)

        ax.fill_between(xs, mu - err, mu + err, alpha=0.5, color='gray')
        ax.plot(xs, mu)
