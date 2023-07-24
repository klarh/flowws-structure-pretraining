import collections

import flowws
from flowws import Argument as Arg
import freud
import numpy as np


@flowws.add_stage_arguments
class NearestNeighbors(flowws.Stage):
    """Calculate neighbors using a distance criterion"""

    ARGS = [
        Arg('neighbor_count', '-n', int, 16, help='Number of neighbors to find'),
        Arg(
            'center_types',
            '-c',
            [str],
            help='Names of types that will be centers of point clouds',
        ),
        Arg(
            'forced_bond_types',
            '-f',
            [str],
            help=(
                'If given, particles that are within the N nearest '
                'neighbors of center particles will be included'
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
            'max_neighbors', '-x', int, 16, help='Maximum number of neighbors to allow'
        ),
    ]

    def run(self, scope, storage):
        if 'type_name_map' not in scope:
            type_map = scope['type_name_map'] = collections.defaultdict(
                lambda: len(type_map)
            )
        self.type_map = scope['type_name_map']
        center_types = set(self.arguments.get('center_types', []))
        forced_bond_types = set(self.arguments.get('forced_bond_types', []))
        self.center_types = np.array([self.type_map[t] for t in center_types])
        self.forced_bond_types = np.array([self.type_map[t] for t in forced_bond_types])

        neighborhood_size = self.arguments['neighbor_count']
        if forced_bond_types and center_types:
            neighborhood_size = self.arguments['max_neighbors']

        scope['nlist_generator'] = self.get_nlist
        scope['pad_size'] = neighborhood_size
        scope['neighborhood_size'] = neighborhood_size

    def get_nlist(self, box, positions, types=None):
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

        if len(self.center_types):
            if types is None:
                raise ValueError('Types not available, but center_types is set')

            bond_source_types = types[nl.query_point_indices]
            bond_filter = np.isin(bond_source_types, self.center_types)
            nl.filter(bond_filter)

            if len(self.forced_bond_types):
                source_indices = np.where(np.isin(types, self.center_types))[0]
                dest_indices = np.where(np.isin(types, self.forced_bond_types))[0]
                q = freud.locality.AABBQuery(box, positions[dest_indices])
                qr = q.query(
                    positions[source_indices],
                    dict(
                        mode='nearest',
                        num_neighbors=self.arguments['forced_bond_count'],
                        exclude_ii=False,
                    ),
                )
                nl_ab = qr.toNeighborList(sort_by_distance=False)
                bond_source = source_indices[nl_ab.query_point_indices]
                bond_dest = dest_indices[nl_ab.point_indices]

                index_i = np.concatenate([nl.query_point_indices, bond_source])
                index_j = np.concatenate([nl.point_indices, bond_dest])
                ijs = np.unique(np.array([index_i, index_j]).T, axis=0)
                distances = freud.box.Box.from_box(box).compute_distances(
                    positions[ijs[:, 0]], positions[ijs[:, 1]]
                )
                idj = np.array(
                    [ijs[:, 0], distances * 1e6, ijs[:, 1]], dtype=np.int32
                ).T
                idj = np.sort(idj, axis=0)
                nl = freud.locality.NeighborList.from_arrays(
                    len(positions), len(positions), idj[:, 0], idj[:, 2], idj[:, 1]
                )
                local_bond_index = np.arange(len(nl)) - np.repeat(
                    nl.segments, nl.neighbor_counts
                )
                filt = local_bond_index < self.arguments['max_neighbors']
                nl.filter(filt)

        return nl
