import flowws
from flowws import Argument as Arg
import freud
import numpy as np


@flowws.add_stage_arguments
class VoronoiNeighbors(flowws.Stage):
    """Calculate neighbors using a Voronoi tessellation"""

    ARGS = [
        Arg(
            'max_neighbors', '-n', int, 20, help='Maximum number of neighbors to allow'
        ),
    ]

    def run(self, scope, storage):
        self.voronoi = freud.locality.Voronoi()

        scope['nlist_generator'] = self.get_nlist
        scope['pad_size'] = self.arguments['max_neighbors']
        scope['max_neighbors'] = self.arguments['max_neighbors']
        scope['use_bond_weights'] = True

    def get_nlist(self, box, positions):
        maximum_neighbors = self.arguments['max_neighbors']
        self.voronoi.compute((box, positions))
        nl = self.voronoi.nlist
        filt = np.ones(len(nl), dtype=np.bool)
        weights = nl.weights
        for i, (start, count) in enumerate(zip(nl.segments, nl.neighbor_counts)):
            if count > maximum_neighbors:
                i_weights = weights[start : start + count]
                bad_indices = np.argsort(-i_weights)[maximum_neighbors:]
                filt[start + bad_indices] = False

        nl = nl.copy().filter(filt)
        return nl
