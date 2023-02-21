from .GalaCore import GalaCore

import flowws
from flowws import Argument as Arg
from tensorflow import keras


@flowws.add_stage_arguments
class GalaScalarRegressor(GalaCore):
    """Regress one scalar for a given environment using geometric algebra attention"""

    ARGS = GalaCore.ARGS + [
        Arg(
            'drop_geometric_embeddings',
            None,
            bool,
            False,
            help='If True, use vector inputs rather than geometric accumulations around the embedding layer',
        ),
        Arg(
            'transfer_freeze',
            None,
            bool,
            False,
            help='If True, freeze pretrained weights for transfer learning',
        ),
    ]

    def run(self, scope, storage):
        self._init(scope, storage)
        if 'encoded_base' not in scope:
            super().run(scope, storage)

        (last_x, last) = scope['encoded_base']
        embedding = last
        inputs = scope['input_symbol']

        if self.arguments['transfer_freeze']:
            frozen_model = keras.models.Model(inputs, scope['encoded_base'])
            frozen_model.trainable = False
            (last_x, last) = frozen_model(inputs)

        if self.arguments['drop_geometric_embeddings']:
            arg = [last_x, last]
            arg[0] = self.maybe_upcast_vector(scope['input_symbol'][0])
        else:
            arg = self.make_layer_inputs(last_x, last)
        (last, ivs, att) = self.Attention(
            self.make_scorefun(),
            self.make_valuefun(self.n_dim),
            True,
            name='final_attention',
            rank=self.rank,
            join_fun=self.join_fun,
            merge_fun=self.merge_fun,
            invariant_mode=self.invar_mode,
            covariant_mode=self.covar_mode,
            include_normalized_products=self.arguments['include_normalized_products'],
        )(arg, return_invariants=True, return_attention=True)

        last = keras.layers.Dense(1)(last)

        embedding_model = keras.models.Model(inputs, embedding)

        scope['input_symbol'] = inputs
        scope['output'] = last
        scope['model'] = keras.models.Model(inputs, scope['output'])
        scope['embedding_model'] = embedding_model
