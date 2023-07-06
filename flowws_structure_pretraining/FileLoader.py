import collections
import functools
import logging
import os

import flowws
from flowws import Argument as Arg
import garnett
import gtar
import numpy as np

logger = logging.getLogger(__name__)

GARNETT_READERS = dict(
    cif=garnett.reader.CifFileReader,
    dcd=garnett.reader.DCDFileReader,
    gsd=garnett.reader.GSDHoomdFileReader,
    pos=garnett.reader.PosFileReader,
    sqlite=garnett.reader.GetarFileReader,
    tar=garnett.reader.GetarFileReader,
    zip=garnett.reader.GetarFileReader,
)

GARNETT_TEXT_MODES = {'cif', 'pos'}


def trajectory_wrapper(fname):
    suffix = os.path.splitext(fname)[1][1:]
    reader = GARNETT_READERS[suffix]
    mode = 'r' if suffix in GARNETT_TEXT_MODES else 'rb'

    f = open(fname, mode)
    # store opened file handle in the cache in case a reader doesn't
    # keep a reference to keep it alive
    return f, reader().read(f)


def frame_wrapper(fname, trajectory_wrapper, index):
    (_, traj) = trajectory_wrapper(fname)
    return traj[index]


class GTARPropertyLoader:
    shapes = dict(
        force=3,
    )

    def __init__(self, fname):
        self.traj = gtar.GTAR(fname, 'r')
        (_, self.frames) = self.traj.framesWithRecordsNamed('position')
        self.records = {rec.getName(): rec for rec in self.traj.getRecordTypes()}

    def get(self, index, name):
        frame = self.frames[index]
        record = self.records[name]
        result = self.traj.getRecord(record, frame)
        if name in self.shapes:
            result = result.reshape((-1, self.shapes[name]))
        return result


class LazyFrame:
    def __init__(self, frame_cache, key, context, type_map, gtar_cache):
        self.frame_cache = frame_cache
        self.gtar_cache = gtar_cache
        self.key = key
        self.context = dict(context)
        self.type_map = type_map

    def _replace(self, **kwargs):
        if 'context' in kwargs:
            self.context = dict(kwargs.pop('context'))
            if not kwargs:
                return self

        msg = (
            'Overwriting values on lazily-loaded frames is not supported. '
            'Consider creating new trajectories instead if lazy loading '
            'is required.'
        )
        raise RuntimeError(msg)

    @property
    def frame(self):
        return self.frame_cache(*self.key)

    @property
    def forces(self):
        if self.gtar_cache is None:
            return None
        return self.gtar_cache(self.key[0]).get(self.key[2], 'force')

    @property
    def positions(self):
        return self.frame.position

    @property
    def box(self):
        return self.frame.box.get_box_array()

    @property
    def types(self):
        frame = self.frame
        try:
            type_map = np.array([self.type_map[t] for t in frame.types])
        except KeyError:
            msg = (
                'Type(s) "{}" not set in input type_names array. All particle '
                'types must be pre-specified for lazy file loading.'
            ).format([t for t in self.type_map if t not in frame.types])
            raise RuntimeError(msg)
        return (
            type_map[frame.typeid]
            if frame.typeid is not None
            else np.zeros(len(frame.position), dtype=np.int32)
        )


@flowws.add_stage_arguments
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
        Arg(
            'custom_context',
            None,
            [(str, eval)],
            help='Custom (key, value) elements to set the context for all frames',
        ),
        Arg('lazy', '-l', bool, False, help='If True, lazily load frames as needed'),
        Arg(
            'lazy_file_limit',
            None,
            int,
            16,
            help='Maximum number of open files for lazy loading',
        ),
        Arg(
            'lazy_cache_size',
            None,
            int,
            1024,
            help='Limit to number of frames saved in lazily-loaded cache',
        ),
        Arg(
            'merge_type_names',
            '-m',
            bool,
            False,
            help=(
                'If True, dynamically find type name encodings as frames '
                'are parsed. Required for eagerly loading frames with differing '
                'particle type names.'
            ),
        ),
        Arg(
            'discover_type_names',
            '-d',
            bool,
            False,
            help='If True, scan all frames for types and print the result',
        ),
        Arg(
            'load_forces',
            None,
            bool,
            False,
            help='If True, also load per-particle forces',
        ),
    ]

    Frame = collections.namedtuple(
        'Frame', ['positions', 'box', 'types', 'context', 'forces']
    )

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

        custom_context = None
        if self.arguments.get('custom_context', None):
            custom_context = {}
            for (key, val) in self.arguments['custom_context']:
                custom_context[key] = val

        if self.arguments['lazy']:
            self.load_frames_lazily(
                scope, frame_slice, all_frames, max_types, custom_context
            )
        else:
            self.load_frames_eagerly(
                scope, frame_slice, all_frames, max_types, custom_context
            )

    def load_frames_eagerly(
        self, scope, frame_slice, all_frames, max_types, custom_context
    ):
        self.type_map = scope.get(
            'type_name_map', collections.defaultdict(lambda: len(self.type_map))
        )
        found_types = None
        warned_about_types = False

        for fname in self.arguments.get('filenames', []):
            context = dict(source='garnett', fname=fname)

            prop_loader = None
            if self.arguments['load_forces'] and any(
                fname.endswith(suf) for suf in ['.zip', '.tar', '.sqlite']
            ):
                prop_loader = GTARPropertyLoader(fname)

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
                    frame_context = (
                        custom_context if custom_context is not None else context
                    )

                    type_names = tuple(frame.types)
                    type_map = [self.type_map[t] for t in frame.types]
                    if self.arguments['merge_type_names']:
                        type_map = np.array(type_map)
                        types = type_map[types]
                    elif (
                        found_types is not None
                        and found_types != type_names
                        and not warned_about_types
                    ):
                        msg = (
                            'Found type names on frame {} of file {} are '
                            'inconsistent: the types found thus far are {}, '
                            'but this frame\'s types are {}'
                        ).format(i, fname, found_types, frame.types)
                        logger.warning(msg)
                        warned_about_types = True

                    forces = None
                    if prop_loader is not None:
                        forces = prop_loader.get(i, 'force')

                    frame = self.Frame(
                        frame.position,
                        frame.box.get_box_array(),
                        types,
                        dict(frame_context),
                        forces,
                    )
                    found_types = type_names
                    max_types = max(max_types, int(np.max(frame.types)) + 1)
                    all_frames.append(frame)

        if self.arguments['discover_type_names']:
            print('Found type names: {}'.format(list(sorted(self.type_map))))
        if self.arguments['merge_type_names']:
            max_types = max(max_types, len(self.type_map))
        scope['max_types'] = max_types
        scope['type_name_map'] = self.type_map

    def load_frames_lazily(
        self, scope, frame_slice, all_frames, max_types, custom_context
    ):
        try:
            self.type_map = {n: i for (i, n) in enumerate(scope['type_names'])}
        except KeyError:
            msg = '"type_names" must be provided in-scope for lazy loading of frames'
            raise KeyError(msg)
        scope['max_types'] = len(self.type_map)
        scope['lazy_frames'] = True

        self.lazy_files = functools.lru_cache(
            maxsize=self.arguments['lazy_file_limit']
        )(trajectory_wrapper)
        self.lazy_gtar_files = None
        if self.arguments['load_forces']:
            self.lazy_gtar_files = functools.lru_cache(
                maxsize=self.arguments['lazy_file_limit']
            )(GTARPropertyLoader)
        self.lazy_frames = functools.lru_cache(
            maxsize=self.arguments['lazy_cache_size']
        )(frame_wrapper)
        for fname in self.arguments.get('filenames', []):
            context = dict(source='garnett', fname=fname)
            (_, traj) = self.lazy_files(fname)
            indices = list(range(len(traj)))[frame_slice]
            for i in indices:
                context['frame'] = i
                key = (fname, self.lazy_files, i)
                frame = LazyFrame(
                    self.lazy_frames, key, context, self.type_map, self.lazy_gtar_files
                )
                all_frames.append(frame)
        scope['type_name_map'] = self.type_map
