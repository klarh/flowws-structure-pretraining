import collections

import flowws
from flowws import Argument as Arg
import numpy as np


@flowws.add_stage_arguments
class SplitDataset(flowws.Stage):
    ARGS = [
        Arg('seed', '-s', int, 13, help='RNG seed to use'),
        Arg(
            'validation_fraction',
            '-v',
            float,
            0.2,
            help='Fraction of data to use for validation set',
        ),
        Arg(
            'test_fraction',
            '-t',
            float,
            0.1,
            help='Fraction of data to use for test set',
        ),
    ]

    class DataSplit:
        def __init__(self, index, particles):
            self.index = index
            self.particles = particles

    def run(self, scope, storage):
        all_frames = scope['loaded_frames']

        val_fraction = self.arguments['validation_fraction']
        test_fraction = self.arguments['test_fraction']
        split_locations = np.cumsum([0, test_fraction, val_fraction, 1.0]) * len(
            all_frames
        )
        split_locations = np.round(split_locations).astype(np.int32)
        segments = [
            slice(left, right)
            for (left, right) in zip(split_locations, split_locations[1:])
        ]

        rng = np.random.default_rng(self.arguments['seed'])
        frame_indices = rng.permutation(len(all_frames))

        test_frames = frame_indices[segments[0]]
        validation_frames = frame_indices[segments[1]]
        train_frames = frame_indices[segments[2]]

        scope['test_frames'] = [self.DataSplit(f, None) for f in test_frames]
        scope['validation_frames'] = [
            self.DataSplit(f, None) for f in validation_frames
        ]
        scope['train_frames'] = [self.DataSplit(f, None) for f in train_frames]
