from .internal import GradientLayer, SumLayer
from .GalaCore import GalaCore

import flowws
from flowws import Argument as Arg
from tensorflow import keras


@flowws.add_stage_arguments
class GalaPotentialRegressor(GalaCore):
    """Produce rotation-invariant scalars or -equivariant conservative
    vector quantities."""

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
        Arg(
            'predict_energy',
            None,
            bool,
            True,
            help='If True, produce energies and use energy labels',
        ),
        Arg(
            'predict_forces',
            None,
            bool,
            False,
            help='If True, produce forces and use force labels',
        ),
    ]

    def run(self, scope, storage):
        if 'encoded_base' not in scope:
            super().run(scope, storage)

        (last_x, last) = scope['encoded_base']
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
        last_x = self.maybe_downcast_vector(last_x)

        last = keras.layers.Dense(self.dilation_dim, name='final_mlp')(last)
        last = self.activation_layer()(last)
        last = keras.layers.Dense(1, name='energy_projection', use_bias=False)(last)
        last = SumLayer()(last)
        energy_prediction = last
        force_prediction = GradientLayer()((last, inputs[0]))

        outputs = []
        if self.arguments['predict_energy']:
            outputs.append(energy_prediction)
        if self.arguments['predict_forces']:
            outputs.append(force_prediction)
        if len(outputs) == 1:
            outputs = outputs[0]
        elif not outputs:
            raise ValueError('No outputs to predict!')

        scope['loss'] = 'mse'
        scope['input_symbol'] = inputs
        scope['output'] = outputs
        scope['model'] = keras.models.Model(inputs, scope['output'])
