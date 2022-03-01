import flowws
from flowws import Argument as Arg
import freud


@flowws.add_stage_arguments
class NearestNeighbors(flowws.Stage):
    """Calculate neighbors using a distance criterion"""

    ARGS = [
        Arg('neighbor_count', '-n', int, 16, help='Number of neighbors to find'),
    ]

    def run(self, scope, storage):
        scope['nlist_generator'] = self.get_nlist
        scope['pad_size'] = self.arguments['neighbor_count']
        scope['neighborhood_size'] = self.arguments['neighbor_count']

    def get_nlist(self, box, positions):
        q = freud.locality.AABBQuery(box, positions)
        qr = q.query(
            positions,
            dict(
                mode='nearest',
                num_neighbors=self.arguments['neighbor_count'],
                exclude_ii=True,
            ),
        )
        nl = qr.toNeighborList(sort_by_distance=False)
        return nl
