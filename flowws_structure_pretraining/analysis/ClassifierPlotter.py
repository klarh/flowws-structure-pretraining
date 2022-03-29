import collections
import functools

import flowws
from flowws import Argument as Arg
import matplotlib
import numpy as np

from ..internal import Remap


@flowws.add_stage_arguments
class ClassifierPlotter(flowws.Stage):
    """Plot classification fractions of observations for each input frame"""

    ARGS = [
        Arg('batch_size', '-b', int, 32, help='Batch size of calculations'),
    ]

    def run(self, scope, storage):
        self.scope = scope
        scope.update(self.value_dict)

        found_key_values = collections.defaultdict(set)
        classifier_contexts = [dict(d) for d in scope['classifier_contexts']]
        for d in classifier_contexts:
            for (k, v) in d.items():
                found_key_values[k].add(v)
        to_remove = [k for (k, vs) in found_key_values.items() if len(vs) == 1]
        for d in classifier_contexts:
            for k in to_remove:
                d.pop(k, None)

        remap = Remap()
        unique_contexts = set(map(lambda x: frozenset(x.items()), classifier_contexts))
        context_sort = lambda v: tuple(sorted(v))
        for v in sorted(unique_contexts, key=context_sort):
            remap(v)
        contexts = np.array([remap(frozenset(d.items())) for d in classifier_contexts])
        contexts = np.repeat(contexts, scope['neighbor_counts'])

        classes = scope['classes']
        sortidx = np.argsort(contexts)

        self.classes = classes[sortidx]
        self.contexts = contexts[sortidx]

        indices = self.classes + self.contexts * self.num_classes
        self.histogram = np.bincount(
            indices, minlength=(np.max(self.contexts) + 1) * self.num_classes
        )
        # (context, class)
        self.histogram = self.histogram.reshape((-1, self.num_classes))
        scope['class_histograms'] = self.histogram

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
        counts = []
        ctxs = []
        for (x, y, ctx) in scope['data_generator']:
            probas = model.predict_on_batch(x)
            pred = np.argmax(probas, axis=-1)
            filt = np.any(x[0] != 0, axis=-1)
            neighbor_count = np.sum(filt, axis=-1)
            xs.append(pred[filt])
            counts.append(neighbor_count)
            ctxs.extend(ctx)

        self.num_classes = probas.shape[-1]
        result['classes'] = np.concatenate(xs, axis=0)
        result['neighbor_counts'] = np.concatenate(counts)
        result['classifier_contexts'] = ctxs
        result['neighbor_segments'] = np.cumsum(
            np.insert(result['neighbor_counts'], 0, 0)
        )[:-1]
        return result

    def evaluate(self, model, scope):
        result = {}
        xs = []
        counts = []
        preds = []
        (rs, ts) = scope['x_train']
        for i_start in range(0, len(rs), self.arguments['batch_size']):
            batch = slice(i_start, i_start + self.arguments['batch_size'])
            x = rs[batch], ts[batch]
            probas = model.predict_on_batch(x)
            pred = np.argmax(probas, axis=-1)
            filt = np.any(x[0] != 0, axis=-1)
            neighbor_count = np.sum(filt, axis=-1)
            xs.append(pred[filt])
            counts.append(neighbor_count)

        self.num_classes = probas.shape[-1]
        result['classes'] = np.concatenate(xs, axis=0)
        result['neighbor_counts'] = np.concatenate(counts)
        result['classifier_contexts'] = scope['x_contexts']
        result['neighbor_segments'] = np.cumsum(
            np.insert(result['neighbor_counts'], 0, 0)
        )[:-1]
        return result

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()

        img = self.histogram.copy().astype(np.float32)
        # normalize
        img /= np.sum(img, axis=-1, keepdims=True)

        imshow = ax.imshow(img.T)
        cbar = fig.colorbar(imshow)
        cbar.solids.set(alpha=1)
