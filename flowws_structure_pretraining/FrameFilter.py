import flowws
from flowws import Argument as Arg
import freud
import numpy as np


@flowws.add_stage_arguments
class FrameFilter(flowws.Stage):
    """Filter particles from loaded frames."""

    ARGS = [
        Arg('whitelist_types', '-w', [str], help='Types that are always included'),
        Arg(
            'forced_bond_types',
            '-f',
            [str],
            help=(
                'If given, particles that are within the N nearest '
                'neighbors of whitelisted particles will be included'
            ),
        ),
        Arg(
            'forced_bond_count',
            None,
            int,
            6,
            help='Number of bonds that will be forced',
        ),
        Arg(
            'skip_empty_whitelist',
            None,
            bool,
            True,
            help='If given and no whitelisted particles are found, return the frame unchanged',
        ),
    ]

    def run(self, scope, storage):
        self.type_map = scope['type_name_map']

        whitelist_types = set(self.arguments['whitelist_types'])
        assert whitelist_types, 'whitelist_types must be provided'
        forced_bond_types = set(self.arguments['forced_bond_types'])
        self.whitelist_types = np.array([self.type_map[t] for t in whitelist_types])
        self.forced_bond_types = np.array([self.type_map[t] for t in forced_bond_types])

        frames = []
        for frame in scope['loaded_frames']:
            frames.append(self.filter(frame))
        scope['loaded_frames'] = frames

    def filter(self, frame):
        final_filter = np.isin(frame.types, self.whitelist_types)

        if not np.any(final_filter) and self.arguments['skip_empty_whitelist']:
            return frame

        if len(self.forced_bond_types):
            dest_indices = np.where(np.isin(frame.types, self.forced_bond_types))[0]
            fbox = freud.box.Box.from_box(frame.box)
            q = freud.locality.AABBQuery(fbox, frame.positions[final_filter])
            qr = q.query(
                frame.positions[dest_indices],
                dict(
                    mode='nearest',
                    num_neighbors=self.arguments['forced_bond_count'],
                    exclude_ii=False,
                ),
            )
            nl = qr.toNeighborList(sort_by_distance=False)
            forced_indices = dest_indices[np.unique(nl.point_indices)]
            final_filter[forced_indices] = True

        return frame._replace(
            positions=frame.positions[final_filter], types=frame.types[final_filter]
        )
