import flowws
from flowws import Argument as Arg
import gtar
import numpy as np
import plato

from ..FileLoader import FileLoader
from ..LoadModel import LoadModel
from .EmbeddingDistance import EmbeddingDistance
from .internal import percentile


@flowws.add_stage_arguments
class EmbeddingDistanceTrajectory(flowws.Stage):
    ARGS = [
        Arg('in_filename', '-i', str, help='Input trajectory filename'),
        Arg('frame_start', None, int, 0, help='Beginning input frame to copy'),
        Arg('frame_end', None, int, help='Final input frame to copy (exclusive)'),
        Arg('frame_skip', None, int, help='Frame selection stride'),
        Arg('out_filename', '-o', str, help='Output dump filename'),
        Arg(
            'color_mode', '-c', str, 'linear', help='Colormap mode (percentile/linear)'
        ),
        Arg('color_min', None, float, 0.0, help='Minimum colormap value'),
        Arg('color_max', None, float, 1.0, help='Maximum colormap value'),
        Arg('contrast', None, float, 1.0, help='Contrast scale'),
        Arg(
            'reverse',
            '-r',
            bool,
            False,
            help='If True, reverse colormap',
        ),
    ] + EmbeddingDistance.ARGS

    def run(self, scope, storage):
        selected_frames = range(
            self.arguments['frame_start'],
            self.arguments.get('frame_end', None),
            self.arguments.get('frame_skip', 1),
        )

        loader_args = dict(
            clear=1, filenames=[self.arguments['in_filename']], frame_skip=1
        )
        embedding_args = {
            k: v
            for (k, v) in self.arguments.items()
            if k in EmbeddingDistance().arg_specifications
        }
        out_filename = self.arguments['out_filename']

        with gtar.GTAR(out_filename, 'w') as traj:
            for frame in selected_frames:
                child_scope = dict(scope)
                loader_args['frame_start'] = frame
                loader_args['frame_end'] = frame + 1
                FileLoader(**loader_args).run(child_scope, storage)
                LoadModel(disable_shuffle=True, subsample=1, no_model=1).run(
                    child_scope, storage
                )
                EmbeddingDistance(**embedding_args).run(child_scope, storage)

                distance = child_scope['reference_distances']
                if not child_scope.get('per_cloud', False):
                    counts = child_scope['neighbor_counts']
                    segments = child_scope['neighbor_segments']

                    averages = np.add.reduceat(distance, segments)
                    averages[counts == 0] = 0
                    averages /= np.clip(counts, 1, np.inf)
                    distance = averages

                frame_data = child_scope['loaded_frames'][0]

                traj.writePath(
                    'frames/{}/position.f32.ind'.format(frame), frame_data.positions
                )
                traj.writePath(
                    'frames/{}/embedding_distance.f32.ind'.format(frame), distance
                )
                traj.writePath('frames/{}/box.f32.uni'.format(frame), frame_data.box)

                for k in ['metrics', 'visuals']:
                    child_scope[k].clear()

        all_distances = []
        frames = []
        frame_means = []
        frame_stds = []
        with gtar.GTAR(out_filename, 'a') as traj:
            for (frame, value) in traj.recordsNamed('embedding_distance'):
                frames.append(frame)
                all_distances.append(value)

            counts = [len(dist) for dist in all_distances]
            distances = self.get_color_map(np.concatenate(all_distances))

            for (frame, count) in zip(frames, counts):
                values = distances[:count]
                distances = distances[count:]
                colors = plato.cmap.cubehelix(values)
                traj.writePath('frames/{}/color.f32.ind'.format(frame), colors)
                frame_means.append(np.mean(values))
                frame_stds.append(np.std(values))

            traj.writePath('vars/embedding_distance_mean.f32.uni/0', frame_means)
            traj.writePath('vars/embedding_distance_std.f32.uni/0', frame_stds)

            assert len(distances) == 0

    def get_color_map(self, distances):
        color_mode = self.arguments['color_mode']
        if color_mode == 'linear':
            mu = np.mean(distances)
            sigma = np.std(distances)
            distances = (distances - mu) / sigma + 0.5
        elif color_mode == 'percentile':
            distances = percentile(distances)
        else:
            raise NotImplementedError(color_mode)

        if not self.arguments['reverse']:
            distances = 1 - distances

        mu = np.mean(distances)
        distances = (distances - mu) * self.arguments['contrast'] + mu

        distances = np.clip(
            distances, self.arguments['color_min'], self.arguments['color_max']
        )

        return distances
