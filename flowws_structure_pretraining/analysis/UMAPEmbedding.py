import flowws
from flowws import Argument as Arg
import umap


@flowws.add_stage_arguments
class UMAPEmbedding(flowws.Stage):
    """Use UMAP to project the embedding"""

    ARGS = [
        Arg('n_neighbors', '-n', int, 15, help='Number of neighbors to consider'),
        Arg('n_components', '-c', int, 2, help='Number of components to produce'),
        Arg(
            'min_dist',
            '-d',
            float,
            0.1,
            help='Minimum distance in resulting projection',
        ),
    ]

    def run(self, scope, storage):
        x = scope['embedding']
        umap_ = scope['umap'] = umap.UMAP(
            n_neighbors=self.arguments['n_neighbors'],
            n_components=self.arguments['n_components'],
            min_dist=self.arguments['min_dist'],
        )
        y = umap_.fit_transform(x)
        scope['embedding_umap'] = y
