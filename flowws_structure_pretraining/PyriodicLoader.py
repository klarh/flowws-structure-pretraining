import collections

import flowws
from flowws import Argument as Arg
import numpy as np
import pyriodic


@flowws.add_stage_arguments
class PyriodicLoader(flowws.Stage):
    """Load frames from the pyriodic database"""

    ARGS = [
        Arg('structures', '-s', [str], help='Names of structures to load'),
        Arg('size', '-z', int, 4096, help='Minimum size to replicate unit cells up to'),
        Arg(
            'noise',
            '-n',
            [float],
            [1e-2, 5e-2, 0.1],
            help='Magnitudes of random noise to add to coordinates',
        ),
        Arg(
            'custom_context',
            None,
            [(str, eval)],
            help='Custom (key, value) elements to set the context for all frames',
        ),
    ]

    Frame = collections.namedtuple('Frame', ['positions', 'box', 'types', 'context'])

    def run(self, scope, storage):
        all_frames = scope.setdefault('loaded_frames', [])
        max_types = 0

        custom_context = None
        if self.arguments.get('custom_context', None):
            custom_context = {}
            for (key, val) in self.arguments['custom_context']:
                custom_context[key] = val

        for name in self.arguments['structures']:
            for (src_frame,) in pyriodic.db.query(
                'select structure from unit_cells where name = ? limit 1', (name,)
            ):
                for noise in self.arguments['noise']:
                    frame = src_frame.rescale_shortest_distance(1.0)
                    frame = frame.replicate_upto(self.arguments['size'])
                    frame = frame.add_gaussian_noise(noise)

                    if custom_context is not None:
                        context = custom_context
                    else:
                        context = dict(
                            source='pyriodic',
                            structure=name,
                            noise=noise,
                        )
                    all_frames.append(
                        self.Frame(frame.positions, frame.box, frame.types, context)
                    )
                    max_types = max(max_types, int(np.max(frame.types)) + 1)

        scope['max_types'] = max_types
