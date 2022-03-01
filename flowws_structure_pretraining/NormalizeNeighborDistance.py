import flowws
from flowws import Argument as Arg
import numpy as np


@flowws.add_stage_arguments
class NormalizeNeighborDistance(flowws.Stage):
    """Normalize the lengthscale of each loaded system
    based on the selected neighbor list calculation"""

    ARGS = [
        Arg('mode', '-m', str, 'mean', help='Averaging mode to use'),
    ]

    def run(self, scope, storage):
        mode = self.arguments['mode']
        nlist_generator = scope['nlist_generator']

        new_frames = []
        for frame in scope['loaded_frames']:
            nl = nlist_generator(frame.box, frame.positions)
            distances = nl.distances

            if mode == 'mean':
                scale = np.mean(distances)
            elif mode == 'median':
                scale = np.median(distances)
            elif mode == 'min':
                scale = np.min(distances)
            else:
                raise NotImplementedError(mode)

            box = np.array(frame.box)
            box[:3] /= scale
            positions = frame.positions / scale
            new_frames.append(frame._replace(box=box, positions=positions))

        scope['loaded_frames'] = new_frames
