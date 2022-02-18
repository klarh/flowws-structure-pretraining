import collections

import flowws
from flowws import Argument as Arg
import freud
import numpy as np


class SANN:
    def __init__(self, system, r_guess=2.0, r_scale=1.25, ball_count=4):
        self.system = system
        self.r_guess = r_guess
        self.r_scale = r_scale
        self.ball_count = ball_count

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, value):
        self._system = value
        self._nq = freud.locality.AABBQuery(self.system.box, self.system.positions)

    def compute(self, query_points):
        done = False
        r_guess = self.r_guess
        r_max = np.min(self.system.box[:3]) / 2
        total_checks = 0
        clipped_checks = 0
        while not done:
            if total_checks < self.ball_count:
                qargs = dict(mode='ball', r_max=r_guess, exclude_ii=True)
            else:
                N = 16
                for _ in range(total_checks):
                    N = max(N + 1, int(self.r_scale * N))
                qargs = dict(
                    mode='nearest', r_guess=r_guess, num_neighbors=N, exclude_ii=True
                )

            q = self._nq.query(query_points, qargs)
            nl = q.toNeighborList(sort_by_distance=True)
            (done, result) = self.create_neighbor_list(nl)

            r_guess *= self.r_scale
            total_checks += 1
            if r_guess > r_max:
                if clipped_checks:
                    raise ValueError('Can\'t find enough neighbors in box')
                clipped_checks += 1
                r_guess = r_max * 0.999
        return result

    def create_neighbor_list(self, nl):
        all_i_s = nl.query_point_indices
        all_j_s = nl.point_indices
        all_d_s = nl.distances
        segments = nl.segments
        counts = nl.neighbor_counts

        if np.any(counts < 3):
            return (False, None)

        cumulative_ds = np.cumsum(all_d_s)
        same_i = all_i_s[1:] == all_i_s[:-1]
        ds_to_smear = cumulative_ds[:-1][~same_i]
        ds_to_smear = np.insert(ds_to_smear, 0, 0)
        cumulative_ds -= np.repeat(ds_to_smear, counts)

        cumulative_sames = np.cumsum(np.insert(same_i, 0, True))
        sames_to_smear = cumulative_sames[:-1][~same_i]
        sames_to_smear = np.insert(sames_to_smear, 0, 1)
        cumulative_sames -= np.repeat(sames_to_smear, counts)
        m = cumulative_sames + 1

        R = cumulative_ds / np.clip(m - 2, 1, 1e30)
        filt = R >= all_d_s
        filt[segments] = True

        if np.all(np.add.reduceat(filt, segments) < counts):
            return (True, nl.copy().filter(filt))
        return (False, None)


@flowws.add_stage_arguments
class SANNeighbors(flowws.Stage):
    """Calculate neighbors using the solid angle nearest neighbors algorithm

    https://aip.scitation.org/doi/10.1063/1.4729313
    """

    ARGS = []

    System = collections.namedtuple('System', ['box', 'positions'])

    def run(self, scope, storage):
        scope['nlist_generator'] = self.get_nlist

    def get_nlist(self, box, positions):
        system = self.System(box, positions)
        sann = SANN(system)
        return sann.compute(positions)
