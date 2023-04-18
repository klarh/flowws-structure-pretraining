import collections
import functools
import itertools
import numpy as np
import rowan
import re


def filter_discrete(x, values=None, delta=1e-2):
    """Remove duplicate points in a point cloud by discretizing by a given lengthscale.

    :param x: Input coordinate array
    :param values: Input value array (integer values associated with each point in x)
    :param delta: Lengthscale controlling how close points must be to be considered duplicates

    """
    result = np.round(x / delta).astype(np.int64)
    if values is not None:
        if values.ndim < 2:
            values = np.atleast_1d(values)[:, None]
        result = np.concatenate([result, values], axis=-1)

    (result, idx) = np.unique(result, axis=0, return_index=True)
    if values is None:
        return x[idx]
    return x[idx], result[:, 3:]


class PointGroup:
    """Wrapper class to handle point-group symmetry replication operations."""

    _REGISTERED_GROUPS = {}
    _PARAMETRIC_GROUPS = set()
    _PARAMETRIC_PATTERNS = {}

    @classmethod
    def register(cls, name):
        def decorator(f):
            cls._REGISTERED_GROUPS[name] = f

            if 'n' in name:
                cls._PARAMETRIC_GROUPS.add(name)
                pattern = name.replace('n', r'(?P<n>\d+)')
                pattern = '^{}$'.format(pattern)
                cls._PARAMETRIC_PATTERNS[pattern] = name
            return f

        return decorator

    @classmethod
    def get(cls, name):
        if name in cls._REGISTERED_GROUPS:
            return cls._REGISTERED_GROUPS[name]
        elif re.match(r'S\d+', name):
            two_n = int(name[1:])
            assert two_n % 2 == 0
            n = two_n // 2
            return functools.partial(cls._REGISTERED_GROUPS['S2n'], n=n)
        else:
            for (pat, base) in cls._PARAMETRIC_PATTERNS.items():
                match = re.match(pat, name)
                if match is not None:
                    kwargs = dict(n=int(match.group('n')))
                    fun = cls._REGISTERED_GROUPS[base]
                    return functools.partial(fun, **kwargs)


@PointGroup.register('Ci')
def inversion(x):
    return np.concatenate([x, -x], axis=0)


@PointGroup.register('Cs')
def reflection(x):
    return rotation_mirrored(x, 1)


@PointGroup.register('Cn')
def nfold_rotation(x, n, axis=(0, 0, 1.0)):
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    quats = rowan.from_axis_angle([axis], thetas)
    return rowan.rotate(quats[:, None], x[None]).reshape((-1, 3))


@PointGroup.register('S2n')
def rotoreflection(x, n, axis=(0, 0, 1.0)):
    theta_flip = np.pi / n
    quat = rowan.from_axis_angle(axis, theta_flip)
    x = np.concatenate([x, rowan.rotate(quat, x) * (1, 1, -1)], axis=0)
    return nfold_rotation(x, n)


@PointGroup.register('Cnh')
def rotation_mirrored(x, n):
    mirror = np.concatenate([x, x * (1, 1, -1)], axis=0)
    return nfold_rotation(mirror, n)


@PointGroup.register('Cnv')
def rotation_vertical_mirrored(x, n):
    thetas = np.linspace(0, np.pi, n, endpoint=False)
    xyzs = [np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)]
    quats = rowan.from_mirror_plane(*xyzs)
    result = rowan.reflect(quats[:, None], x[None]).reshape((-1, 3))
    result = np.concatenate([result, nfold_rotation(x, n)], axis=0)
    return result


@PointGroup.register('Dn')
def dihedral(x, n, dihedral_axis=(1.0, 0, 0), nfold_axis=(0, 0, 1.0)):
    theta_flip = np.pi
    quat = rowan.from_axis_angle(dihedral_axis, theta_flip)
    x = np.concatenate([x, rowan.rotate(quat, x)], axis=0)
    return nfold_rotation(x, n, nfold_axis)


@PointGroup.register('Dnd')
def antiprismatic(x, n):
    result = rotation_vertical_mirrored(x, n)
    theta_flip = np.pi / n
    quat = rowan.from_axis_angle([0, 0, 1], theta_flip)
    result = np.concatenate([result, rowan.rotate(quat, result) * (1, 1, -1)], axis=0)
    return result


@PointGroup.register('Dnh')
def prismatic(x, n):
    result = rotation_vertical_mirrored(x, n)
    result = np.concatenate([result, result * (1, 1, -1)], axis=0)
    return result


@PointGroup.register('T')
def chiral_tetrahedral(x):
    pieces = [x]
    for corner in [(1, 1, 1), (1, -1, -1), (-1, -1, 1), (-1, 1, -1)]:
        pieces.append(nfold_rotation(x, 3, axis=corner)[len(x) :])

    for face in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        pieces.append(nfold_rotation(x, 2, axis=face)[len(x) :])
    return np.concatenate(pieces, axis=0)


@PointGroup.register('Td')
def full_tetrahedral(x):
    pieces = [x]
    for corner in [(1, 1, 1), (1, -1, -1), (-1, -1, 1), (-1, 1, -1)]:
        pieces.append(nfold_rotation(x, 3, axis=corner)[len(x) :])

    for face in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        pieces.append(rotoreflection(x, 2, axis=face)[len(x) :])

    x = np.concatenate(pieces, axis=0)
    pieces = []

    f = 1.0 / np.sqrt(2)
    mirrors = np.array(
        [
            (f, f, 0),
            (0, f, f),
            (f, 0, f),
        ]
    )
    quats = rowan.from_mirror_plane(*mirrors.T)
    for mirror in quats:
        pieces.append(rowan.reflect(mirror[None], x))

    return np.concatenate(pieces, axis=0)


@PointGroup.register('Th')
def pyritohedral(x):
    pieces = [x]
    for corner in [(1, 1, 1), (1, -1, -1), (-1, -1, 1), (-1, 1, -1)]:
        pieces.append(rotoreflection(x, 3, axis=corner)[len(x) :])

    for face in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        pieces.append(nfold_rotation(x, 2, axis=face)[len(x) :])

    mirrors = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
        ]
    )
    quats = rowan.from_mirror_plane(*mirrors.T)
    for mirror in quats:
        pieces.append(rowan.reflect(mirror[None], x))
    return inversion(np.concatenate(pieces, axis=0))


@PointGroup.register('Th')
def pyritohedral(x):
    x = chiral_tetrahedral(x)
    pieces = []

    mirrors = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
        ]
    )
    quats = rowan.from_mirror_plane(*mirrors.T)
    for mirror in quats:
        pieces.append(rowan.reflect(mirror[None], x))
    return inversion(np.concatenate(pieces, axis=0))


@PointGroup.register('O')
def chiral_octahedral(x):
    pieces = [x]
    for corner in [(1, 1, 1), (1, -1, -1), (-1, -1, 1), (-1, 1, -1)]:
        pieces.append(nfold_rotation(x, 3, axis=corner)[len(x) :])

    for face in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        pieces.append(nfold_rotation(x, 4, axis=face)[len(x) :])

    for i, edge in itertools.product(range(3), [(0.5, 0.5, 0), (0.5, -0.5, 0)]):
        edge = np.roll(edge, i)
        pieces.append(nfold_rotation(x, 2, axis=edge)[len(x) :])
    return np.concatenate(pieces, axis=0)


@PointGroup.register('Oh')
def full_octahedral(x):
    x = chiral_octahedral(x)
    pieces = [x]

    f = 1.0 / np.sqrt(2)
    mirrors = np.array(
        [
            (f, f, 0),
            (0, f, f),
            (f, 0, f),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
        ]
    )
    quats = rowan.from_mirror_plane(*mirrors.T)
    for mirror in quats:
        pieces.append(rowan.reflect(mirror[None], x))
    return np.concatenate(pieces, axis=0)


IcosahedralSymmetries = collections.namedtuple(
    'IcosahedralSymmetries', ['d3_axes', 'd5_axes', 'mirror_planes']
)


@functools.lru_cache
def get_icosahedral_symmetries():
    import plato

    phi = (1 + np.sqrt(5)) / 2
    coords = []
    for i, one, phi in itertools.product(range(3), [-1, 1], [-phi, phi]):
        coords.append(np.roll((0, one, phi), i))
    coords = np.array(coords)
    np.random.seed(13)
    v = np.random.normal(size=3)
    (verts, faces) = plato.geometry.convexHull(coords)
    norms = []
    edge_map = {}
    for face in faces:
        (a, b, c) = face[:3]
        edge_map[a] = edge_map[b] = c
        edge_map[c] = a
        norm = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        ortho = np.cross(norm, verts[a])
        norms.append((norm, ortho))
    norms = np.array(norms)
    d3_axes = norms[np.dot(norms[:, 0, :], v) >= 0]
    d3_axes /= np.linalg.norm(d3_axes, axis=-1, keepdims=True)

    d5_index = np.where(np.dot(coords, v) >= 0)[0]
    d5_axes = coords[np.dot(coords, v) >= 0]
    d5_axes = []
    for i in d5_index:
        norm = coords[i]
        ortho = np.cross(norm, coords[edge_map[i]])
        d5_axes.append((norm, ortho))
    d5_axes = np.array(d5_axes)
    d5_axes /= np.linalg.norm(d5_axes, axis=-1, keepdims=True)

    decomp = plato.geometry.convexDecomposition(verts)
    edge_indices = np.array(list(decomp.edges))
    edge_vectors = verts[edge_indices[:, 1]] - verts[edge_indices[:, 0]]
    dots = np.abs(np.sum(edge_vectors[:, None] * edge_vectors[None], axis=-1)) / np.sum(
        edge_vectors**2, axis=-1
    )
    par_i, par_j = np.where(dots > 1 - 1e-3)
    filt = par_i != par_j
    par_i, par_j = par_i[filt], par_j[filt]
    good_edge_pairs = []
    mirror_planes = []
    for (i, j) in zip(par_i, par_j):
        (a, b), (c, d) = edge_indices[i], edge_indices[j]
        norm = np.cross(verts[c] - verts[a], verts[d] - verts[a])
        mirror_planes.append(norm)
    mirror_planes = np.array(mirror_planes)
    mirror_planes = mirror_planes[np.dot(mirror_planes, v) >= 0]
    return IcosahedralSymmetries(d3_axes, d5_axes, mirror_planes)


@PointGroup.register('I')
def chiral_icosahedral(x):
    sym = get_icosahedral_symmetries()
    pieces = [x]
    for (ax, dax) in sym.d3_axes:
        pieces.append(dihedral(x, 3, dihedral_axis=dax, nfold_axis=ax))

    pieces = np.concatenate(pieces, axis=0)
    x = pieces
    pieces = [x]

    for (ax, dax) in sym.d5_axes:
        pieces.append(dihedral(x, 5, dihedral_axis=dax, nfold_axis=ax))

    pieces = np.concatenate(pieces, axis=0)
    x = pieces
    pieces = [x]

    quats = rowan.from_mirror_plane(*np.array(sym.mirror_planes).T)
    for mirror in quats:
        break
        pieces.append(rowan.reflect(mirror[None], x))
    return np.concatenate(pieces, axis=0)


@PointGroup.register('Ih')
def full_icosahedral(x):
    sym = get_icosahedral_symmetries()
    pieces = [x]
    for (ax, _) in sym.d3_axes:
        pieces.append(nfold_rotation(x, 3, axis=ax))

    pieces = np.concatenate(pieces, axis=0)
    x = pieces
    pieces = [x]

    for (ax, _) in sym.d5_axes:
        pieces.append(nfold_rotation(x, 5, axis=ax))

    pieces = np.concatenate(pieces, axis=0)
    x = pieces
    pieces = [x]

    quats = rowan.from_mirror_plane(*np.array(sym.mirror_planes).T)
    for mirror in quats:
        pieces.append(rowan.reflect(mirror[None], x))
    return np.concatenate(pieces, axis=0)
