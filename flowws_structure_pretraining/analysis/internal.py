import collections
import functools

import numpy as np


def percentile(x):
    sortidx = np.argsort(x)
    result = np.empty(len(x))
    result[sortidx] = np.linspace(0, 1, len(sortidx))
    return result


class CumulativePercentile:
    def __init__(self):
        self.memory = {}

    def __call__(self, key, values):
        self.memory[key] = values

        all_values = np.concatenate(list(self.memory.values()))
        sorted_all_values = np.sort(all_values)
        indices = np.searchsorted(sorted_all_values, values)
        return np.linspace(0, 1, len(sorted_all_values))[indices]


class GeneratorVisualizer:

    Prediction = collections.namedtuple(
        'Prediction', ['positions', 'starts', 'ends', 'theta']
    )

    @functools.lru_cache(maxsize=1)
    def get_predictions(self, cache_key):
        scope = self.scope
        frame = self.Frame(
            scope['position'], scope['box'], scope['type'], dict(type='dynamic')
        )

        workflow = scope['preprocess_workflow']
        workflow.scope.update(self.scope)
        workflow.scope['loaded_frames'] = [frame]
        child_scope = workflow.run()
        scope['x_scale'] = child_scope['x_scale']

        model = scope['model']
        predictions = []
        counts = []
        rijs = []
        for (x, _, _) in child_scope['data_generator']:
            pred = model.predict_on_batch(x)
            filt = np.any(x[0] != 0, axis=-1)
            rijs.append(x[0][filt])
            neighbor_count = np.sum(filt, axis=-1)
            predictions.append(self.filter_prediction(pred, filt))
            counts.append(neighbor_count)

        rijs = np.concatenate(rijs, axis=0) * child_scope['x_scale']
        predictions = np.concatenate(predictions, axis=0)
        counts = np.concatenate(counts, axis=0)

        positions = scope['position']
        starts = np.repeat(positions, counts, axis=0)
        ends = starts + rijs
        prediction = self.Prediction(positions, starts, ends, predictions)
        prediction = self.theta_prediction(prediction)

        return prediction

    def remap_theta(self, theta):
        rescale = False

        if 'mode' not in self.arguments or not self.arguments['mode']:
            rescale = True
        elif self.arguments['mode'] == 'noop':
            rescale = False
        elif self.arguments['mode'] == 'logits':
            theta = -np.log(1.0 / theta - 1)
            rescale = True
        elif self.arguments['mode'] == 'percentile':
            theta = percentile(theta)
        elif self.arguments['mode'] == 'cumulative_percentile':
            cp = getattr(self, 'cumulative_percentile', CumulativePercentile())
            self.cumulative_percentile = cp
            theta = cp(self.scope['cache_key'], theta)
        else:
            raise NotImplementedError(self.arguments['mode'])

        print(
            'Theta statistics:',
            np.min(theta),
            np.max(theta),
            np.mean(theta),
            np.std(theta),
        )
        if rescale:
            mu = np.mean(theta)
            sigma = np.std(theta)
            theta = theta - mu
            theta /= sigma
            theta += 0.5

        if self.arguments['reverse']:
            theta = 1 - theta

        if 'contrast' in self.arguments:
            mu = np.mean(theta)
            theta = mu + (theta - mu) * self.arguments['contrast']

        theta = np.clip(theta, 0, 1)
        theta = (1 - theta) * self.arguments['color_min'] + theta * self.arguments[
            'color_max'
        ]
        return theta

    @staticmethod
    def filter_prediction(pred, filt):
        return pred[filt]
