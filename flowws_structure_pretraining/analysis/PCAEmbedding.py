import flowws
from flowws import Argument as Arg
import numpy as np
from sklearn.decomposition import PCA


@flowws.add_stage_arguments
class PCAEmbedding(flowws.Stage):
    """Use PCA to project the embedding"""

    ARGS = [
        Arg('n_dim', '-d', int, 8, help='Target dimensionality'),
    ]

    def run(self, scope, storage):
        x = scope['embedding']
        self.pca = scope['pca'] = PCA(self.arguments['n_dim'])
        y = self.pca.fit_transform(x)
        scope['embedding_pca'] = y
        scope.setdefault('visuals', []).append(self)

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()
        x = 1 + np.arange(self.pca.n_components)
        y = np.cumsum(self.pca.explained_variance_ratio_)
        ax.plot(x, y)
        ax.set_xlabel('Components')
        ax.set_ylabel('Explained variance')
