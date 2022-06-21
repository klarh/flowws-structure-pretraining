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
        Arg(
            'aggregate_probabilities',
            None,
            bool,
            False,
            help='If True, force aggregation of classes by probability',
        ),
        Arg(
            'plot_series',
            '-s',
            bool,
            False,
            help='If True, plot series as lines rather than heatmaps/images',
        ),
    ]

    def run(self, scope, storage):
        self.use_probabilities = self.arguments['aggregate_probabilities']
        self.use_probabilities |= scope.get('multilabel', False)
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
        if not scope.get('per_cloud', False):
            contexts = np.repeat(contexts, scope['neighbor_counts'])

        classes = scope['classes']
        sortidx = np.argsort(contexts)

        self.classes = classes[sortidx]
        self.contexts = contexts[sortidx]

        if self.use_probabilities:
            hist = np.zeros((np.max(self.contexts) + 1, self.num_classes))
            for (i, bins) in zip(self.contexts, self.classes):
                hist[i] += bins
            normalization = np.bincount(self.contexts)
            self.histogram = hist / normalization[:, None]
        else:
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
        neighbor_count = 1
        for (x, y, ctx) in scope['data_generator']:
            probas = model.predict_on_batch(x)
            pred = probas if self.use_probabilities else np.argmax(probas, axis=-1)
            if not scope.get('per_cloud', False):
                filt = np.any(x[0] != 0, axis=-1)
                neighbor_count = np.sum(filt, axis=-1)
                pred = pred[filt]
            xs.append(pred)
            counts.append(neighbor_count)
            ctxs.extend(ctx)

        self.num_classes = probas.shape[-1]
        result['classes'] = np.concatenate(xs, axis=0)
        result['classifier_contexts'] = ctxs
        if not scope.get('per_cloud', False):
            result['neighbor_counts'] = np.concatenate(counts)
            result['neighbor_segments'] = np.cumsum(
                np.insert(result['neighbor_counts'], 0, 0)
            )[:-1]
        return result

    def evaluate(self, model, scope):
        result = {}
        xs = []
        counts = []
        preds = []
        neighbor_count = 1
        (rs, ts) = scope['x_train']
        for i_start in range(0, len(rs), self.arguments['batch_size']):
            batch = slice(i_start, i_start + self.arguments['batch_size'])
            x = rs[batch], ts[batch]
            probas = model.predict_on_batch(x)
            pred = probas if self.use_probabilities else np.argmax(probas, axis=-1)
            if not scope.get('per_cloud', False):
                filt = np.any(x[0] != 0, axis=-1)
                neighbor_count = np.sum(filt, axis=-1)
                pred = pred[filt]
            xs.append(pred)
            counts.append(neighbor_count)

        self.num_classes = probas.shape[-1]
        result['classes'] = np.concatenate(xs, axis=0)
        result['classifier_contexts'] = scope['x_contexts']
        if not scope.get('per_cloud', False):
            result['neighbor_counts'] = np.concatenate(counts)
            result['neighbor_segments'] = np.cumsum(
                np.insert(result['neighbor_counts'], 0, 0)
            )[:-1]
        return result

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()

        img = self.histogram.copy().astype(np.float32)
        # normalize
        img /= np.sum(img, axis=-1, keepdims=True)

        if self.arguments['plot_series']:
            ax.plot(img)
        else:
            imshow = ax.imshow(img.T)
            cbar = fig.colorbar(imshow)
            cbar.solids.set(alpha=1)
