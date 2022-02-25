import collections

import flowws
from flowws import Argument as Arg
import garnett
import numpy as np


@flowws.add_stage_arguments
@flowws.register_module
class FileLoader(flowws.Stage):
    """Load trajectory files"""

    ARGS = [
        Arg('filenames', '-f', [str], help='Names of files to load'),
        Arg(
            'frame_start',
            None,
            int,
            0,
            help='Number of frames to skip from each trajectory',
        ),
        Arg(
            'frame_end',
            None,
            int,
            help='End frame (exclusive) to take from each trajectory',
        ),
        Arg(
            'frame_skip',
            None,
            int,
            help='Number of frames to skip while traversing trajectory',
        ),
        Arg(
            'clear',
            '-c',
            bool,
            False,
            help='If True, clear the list of loaded files first',
        ),
    ]

    Frame = collections.namedtuple('Frame', ['positions', 'box', 'types', 'context'])

    def run(self, scope, storage):
        frame_slice = slice(
            self.arguments['frame_start'],
            self.arguments.get('frame_end', None),
            self.arguments.get('frame_skip', None),
        )
        all_frames = scope.setdefault('loaded_frames', [])
        max_types = scope.get('max_types', 0)

        if self.arguments['clear']:
            all_frames.clear()

        for fname in self.arguments.get('filenames', []):
            context = dict(source='garnett', fname=fname)
            with garnett.read(fname) as traj:
                indices = list(range(len(traj)))[frame_slice]
                for i in indices:
                    frame = traj[i]
                    context['frame'] = i
                    types = (
                        frame.typeid
                        if frame.typeid is not None
                        else np.zeros(len(frame.position), dtype=np.int32)
                    )
                    frame = self.Frame(
                        frame.position, frame.box.get_box_array(), types, dict(context)
                    )
                    max_types = max(max_types, int(np.max(frame.types)) + 1)
                    all_frames.append(frame)

        scope['max_types'] = max_types
