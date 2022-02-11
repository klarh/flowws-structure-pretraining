import flowws
from flowws import Argument as Arg
import freud
import numpy as np


@flowws.add_stage_arguments
@flowws.register_module
class DistanceNeighbors(flowws.Stage):
    """Calculate neighbors using a distance criterion"""

    ARGS = [
        Arg(
            'max_neighbors', '-n', int, 16, help='Maximum number of neighbors to allow'
        ),
        Arg('r_cut', '-r', float, 2, help='Cutoff distance to use'),
    ]

    def run(self, scope, storage):
        scope['nlist_generator'] = self.get_nlist
        scope['pad_size'] = self.arguments['max_neighbors']
        scope['max_neighbors'] = self.arguments['max_neighbors']

    def get_nlist(self, box, positions):
        q = freud.locality.AABBQuery(box, positions)
        maximum_neighbors = self.arguments['max_neighbors']
        qr = q.query(
            positions, dict(mode='ball', r_max=self.arguments['r_cut'], exclude_ii=True)
        )
        nl = qr.toNeighborList(sort_by_distance=False)

        if np.any(nl.neighbor_counts > maximum_neighbors):
            filt = np.ones(len(nl), dtype=np.bool)
            distances = nl.distances
            for i, (start, count) in enumerate(zip(nl.segments, nl.neighbor_counts)):
                if count > maximum_neighbors:
                    i_weights = distances[start : start + count]
                    bad_indices = np.argsort(i_weights)[maximum_neighbors:]
                    filt[start + bad_indices] = False
            nl = nl.copy().filter(filt)
        return nl
