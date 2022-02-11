import flowws
from flowws import Argument as Arg
from sklearn.decomposition import PCA


@flowws.add_stage_arguments
@flowws.register_module
class PCAEmbedding(flowws.Stage):
    """Use PCA to project the embedding"""

    ARGS = [
        Arg('n_dim', '-d', int, 2, help='Target dimensionality'),
    ]

    def run(self, scope, storage):
        x = scope['embedding']
        shape = list(x.shape[:-1])
        pca = scope['pca'] = PCA(self.arguments['n_dim'])
        y = pca.fit_transform(x.reshape((-1, x.shape[-1])))
        y = y.reshape(shape + [self.arguments['n_dim']])
        scope['embedding_pca'] = y
