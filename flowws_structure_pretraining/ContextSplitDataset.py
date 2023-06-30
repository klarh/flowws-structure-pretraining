import flowws
from flowws import Argument as Arg
import numpy as np


def filter_nested_values(vals, filt):
    if isinstance(vals, list):
        return [filter_nested_values(v, filt) for v in vals]
    elif isinstance(vals, tuple):
        return tuple(filter_nested_values(v, filt) for v in vals)
    else:
        return vals[filt]


@flowws.add_stage_arguments
class ContextSplitDataset(flowws.Stage):
    """Split datasets based on a value in the context dictionary"""

    ARGS = [
        Arg('key', '-k', str, 'dataset', help='Context key to use for splitting'),
    ]

    def run(self, scope, storage):
        x = scope['x_train']
        y = scope['y_train']
        contexts = scope['x_contexts']

        targets = np.array([ctx[self.arguments['key']] for ctx in contexts])

        (dataset_names, dataset_targets) = np.unique(targets, return_inverse=True)

        datasets = {}
        for i, name in enumerate(dataset_names):
            filt = np.where(dataset_targets == i)[0]
            dataset = filter_nested_values((x, y), filt)
            ctx = contexts[filt]

            scope['{}_data'.format(name)] = dataset
            scope['x_{}'.format(name)] = dataset[0]
            scope['y_{}'.format(name)] = dataset[1]
            ctx_name = 'x_contexts' if name == 'train' else '{}_contexts'.format(name)
            scope[ctx_name] = ctx

        scope['dataset_names'] = dataset_names
