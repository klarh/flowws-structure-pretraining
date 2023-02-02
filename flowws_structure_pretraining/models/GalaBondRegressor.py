from .GalaCore import GalaCore

import flowws
from flowws import Argument as Arg
from tensorflow import keras


@flowws.add_stage_arguments
class GalaBondRegressor(GalaCore):
    """Regress one bond for a given environment using geometric algebra attention"""

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
        inputs = scope['input_symbol']

        if self.arguments['transfer_freeze']:
            frozen_model = keras.models.Model(inputs, scope['encoded_base'])
            frozen_model.trainable = False
            (last_x, last) = frozen_model(inputs)

        if 'equivariant_rescale_factor' in scope:
            last_x = last_x / scope['equivariant_rescale_factor']

        if self.arguments['drop_geometric_embeddings']:
            arg = [last_x, last]
            arg[0] = self.maybe_upcast_vector(scope['input_symbol'][0])
            if 'equivariant_rescale_factor' in scope:
                arg[0] = arg[0] / scope['equivariant_rescale_factor']
        else:
            arg = self.make_layer_inputs(last_x, last)
        (last_x, ivs, att) = self.AttentionVector(
            self.make_scorefun(),
            self.make_valuefun(self.n_dim),
            self.make_valuefun(1),
            True,
            name='final_attention',
            rank=self.rank,
            join_fun=self.join_fun,
            merge_fun=self.merge_fun,
            invariant_mode=self.invar_mode,
            covariant_mode=self.covar_mode,
            include_normalized_products=self.arguments['include_normalized_products'],
            convex_covariants=self.arguments['convex_covariants'],
        )(arg, return_invariants=True, return_attention=True)
        last_x = self.maybe_downcast_vector(last_x)

        scope['input_symbol'] = inputs
        scope['output'] = last_x
        scope['model'] = keras.models.Model(inputs, scope['output'])
