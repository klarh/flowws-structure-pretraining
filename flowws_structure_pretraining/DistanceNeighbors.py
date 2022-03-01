import itertools

import flowws
from flowws import Argument as Arg
import freud
import numpy as np


@flowws.add_stage_arguments
class DistanceNeighbors(flowws.Stage):
    """Calculate neighbors using a distance criterion"""

    ARGS = [
        Arg(
            'max_neighbors', '-n', int, 16, help='Maximum number of neighbors to allow'
        ),
        Arg('r_cut', '-r', float, 2, help='Cutoff distance to use'),
        Arg(
            'rdf_shells',
            None,
            int,
            help='Calculate cutoff distance based on the given shell number in the RDF',
        ),
        Arg('rdf_distance', None, float, 0.1, help='Bin size to use for RDFs'),
        Arg(
            'rdf_smoothing',
            None,
            int,
            8,
            help='Smoothing lengthscale to apply for RDFs',
        ),
    ]

    def run(self, scope, storage):
        scope['nlist_generator'] = self.get_nlist
        scope['pad_size'] = self.arguments['max_neighbors']
        scope['max_neighbors'] = self.arguments['max_neighbors']

    def get_nlist(self, box, positions):
        rcut = self.arguments['r_cut']
        if 'rdf_shells' in self.arguments:
            done = False
            rdf_rmax = min(0.499 * np.min(box[:3]), 2 * self.arguments['r_cut'])
            while not done:
                rdf = freud.density.RDF(
                    int(rdf_rmax / self.arguments['rdf_distance']), rdf_rmax
                )
                try:
                    rdf.compute((box, positions))
                except RuntimeError:
                    done = True
                    break

                r = rdf.bin_centers
                y = rdf.rdf
                dr = np.diff(rdf.bounds) / rdf.nbins
                v = np.diff(np.cumsum(r * dr * y))
                dist = self.arguments['rdf_smoothing']
                filt = np.full(dist, 1.0 / dist)
                v = np.convolve(v, filt, mode='same')
                d = np.diff(v)
                s = np.sign(d)
                sd = np.diff(s)
                sd *= np.logical_and(s[:-1] != 0, s[1:] != 0)
                maxima = set(np.where(sd < -0.5)[0] + 1)
                minima = set(np.where(sd > 0.5)[0] + 1)
                extrema = list(sorted(maxima.union(minima)))
                extrema_groups = itertools.groupby(extrema, lambda x: x in maxima)

                take_minimum = False
                shells = []
                for (in_maxima, group) in extrema_groups:
                    if in_maxima:
                        take_minimum = True
                    elif take_minimum:
                        index = int(np.median(list(group)))
                        shells.append(index)
                        take_minimum = False

                if len(shells) >= self.arguments['rdf_shells']:
                    index = shells[self.arguments['rdf_shells'] - 1]
                    rcut = r[index]
                    done = rdf.nbins - index > self.arguments['rdf_smoothing']

                # end early if we can't find enough distinct shells
                done = done or np.sum(v) > 4 * self.arguments['max_neighbors']

                if not done:
                    rdf_rmax *= 1.25

        q = freud.locality.AABBQuery(box, positions)
        maximum_neighbors = self.arguments['max_neighbors']
        qr = q.query(positions, dict(mode='ball', r_max=rcut, exclude_ii=True))
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
