from .internal import GradientLayer, NeighborhoodReduction, ScaledMSELoss, SumLayer
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
        Arg(
            'forces_first',
            None,
            bool,
            False,
            help='If True, output forces first instead of energies',
        ),
        Arg(
            'loss_mixing_beta',
            None,
            float,
            1.0,
            help='MSE loss contribution for second set of predicted quantities',
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
        if scope.get('per_molecule', False):
            energy_prediction = last = NeighborhoodReduction('sum', name='energy')(last)
        total_sum = SumLayer()(last)
        force_prediction = GradientLayer(name='force')((total_sum, inputs[0]))[0]

        outputs = []
        if self.arguments['predict_energy']:
            outputs.append(energy_prediction)
        if self.arguments['predict_forces']:
            outputs.append(force_prediction)
        num_outputs = len(outputs)
        if num_outputs == 1:
            outputs = outputs[0]
        elif num_outputs == 2 and self.arguments['forces_first']:
            outputs = outputs[::-1]
        elif not outputs:
            raise ValueError('No outputs to predict!')

        loss = 'mse'
        if num_outputs == 2:
            loss = ['mse', ScaledMSELoss(self.arguments['loss_mixing_beta'])]

        scope['loss'] = loss
        scope['input_symbol'] = inputs
        scope['output'] = outputs
        scope['model'] = keras.models.Model(inputs, scope['output'])
