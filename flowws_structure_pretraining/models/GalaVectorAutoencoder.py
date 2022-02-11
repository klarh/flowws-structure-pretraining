from .internal import NeighborDistanceNormalization

import flowws
from flowws import Argument as Arg
from geometric_algebra_attention import keras as gala
import numpy as np
from tensorflow import keras

NORMALIZATION_LAYERS = {
    None: lambda _: [],
    'none': lambda _: [],
    'batch': lambda _: [keras.layers.BatchNormalization()],
    'layer': lambda _: [keras.layers.LayerNormalization()],
    'momentum': lambda _: [gala.MomentumNormalization()],
}

NORMALIZATION_LAYER_DOC = ' (any of {})'.format(
    ','.join(map(str, NORMALIZATION_LAYERS))
)


@flowws.add_stage_arguments
@flowws.register_module
class GalaVectorAutoencoder(flowws.Stage):
    """Reproduce input point clouds"""

    ARGS = [
        Arg('n_dim', '-n', int, 32, help='Working dimension of model'),
        Arg('dilation_factor', None, float, 2.0, help='Width factor for MLPs'),
        Arg(
            'block_nonlinearity',
            '-b',
            bool,
            True,
            help='Add a nonlinearity to each block',
        ),
        Arg(
            'residual',
            '-r',
            bool,
            True,
            help='Use residual connections over each block',
        ),
        Arg(
            'join_fun',
            '-j',
            str,
            'concat',
            help='Function to use for joining invariant and node-level signals',
        ),
        Arg(
            'merge_fun',
            '-m',
            str,
            'concat',
            help='Function to use for merging node-level signals',
        ),
        Arg('dropout', '-d', float, 0, help='Dropout probability within network'),
        Arg('num_blocks', None, int, 1, help='Number of blocks to use'),
        Arg(
            'num_vector_blocks',
            None,
            int,
            1,
            help='Number of vector-valued blocks to use',
        ),
        Arg('rank', None, int, 2, help='Attention calculation rank'),
        Arg(
            'activation',
            '-a',
            str,
            'relu',
            help='Activation function to use within network',
        ),
        Arg(
            'normalize_distances',
            None,
            str,
            help='Create scale-invariant networks by normalizing neighbor distances (mean/min)',
        ),
        Arg('invar_mode', '-i', str, 'full', help='Rotation-invariant mode switch'),
        Arg('covar_mode', '-c', str, 'full', help='Rotation-covariant mode switch'),
        Arg(
            'score_normalization',
            None,
            str,
            'layer',
            help=(
                'Normalizations to apply to score (attention) function'
                + NORMALIZATION_LAYER_DOC
            ),
        ),
        Arg(
            'value_normalization',
            None,
            str,
            'layer',
            help=(
                'Normalizations to apply to value function' + NORMALIZATION_LAYER_DOC
            ),
        ),
        Arg(
            'block_normalization',
            None,
            str,
            help=(
                'Normalizations to apply to the output of each attention block'
                + NORMALIZATION_LAYER_DOC
            ),
        ),
        Arg(
            'invariant_value_normalization',
            None,
            str,
            'momentum',
            help=(
                'Normalizations to apply to value function, before MLP layers'
                + NORMALIZATION_LAYER_DOC
            ),
        ),
        Arg(
            'equivariant_value_normalization',
            None,
            str,
            'layer',
            help=(
                'Normalizations to apply to equivariant results'
                + NORMALIZATION_LAYER_DOC
            ),
        ),
        Arg(
            'use_multivectors',
            None,
            bool,
            False,
            help='If True, use multivector intermediates for calculations',
        ),
    ]

    def run(self, scope, storage):
        use_weights = scope.get('use_bond_weights', False)
        n_dim = self.arguments['n_dim']
        dilation = self.arguments['dilation_factor']
        block_nonlin = self.arguments['block_nonlinearity']
        residual = self.arguments['residual']
        join_fun = self.arguments['join_fun']
        merge_fun = self.arguments['merge_fun']
        dropout = self.arguments['dropout']
        num_blocks = self.arguments['num_blocks']
        rank = self.arguments['rank']
        activation = self.arguments['activation']
        distance_norm = self.arguments.get('normalize_distances', None)
        invar_mode = self.arguments['invar_mode']
        covar_mode = self.arguments['covar_mode']

        if self.arguments['use_multivectors']:
            Attention = gala.MultivectorAttention
            AttentionVector = gala.Multivector2MultivectorAttention
            maybe_upcast_vector = gala.Vector2Multivector()
            maybe_downcast_vector = gala.Multivector2Vector()
        else:
            Attention = gala.VectorAttention
            AttentionVector = gala.Vector2VectorAttention
            maybe_upcast_vector = lambda x: x
            maybe_downcast_vector = lambda x: x

        normalization_getter = lambda key: (
            NORMALIZATION_LAYERS[self.arguments.get(key + '_normalization', None)](rank)
        )

        if self.arguments['use_multivectors']:
            Attention = gala.MultivectorAttention
            AttentionVector = gala.Multivector2MultivectorAttention
            maybe_upcast_vector = gala.Vector2Multivector()
            maybe_downcast_vector = gala.Multivector2Vector()
        else:
            Attention = gala.VectorAttention
            AttentionVector = gala.Vector2VectorAttention
            maybe_upcast_vector = lambda x: x
            maybe_downcast_vector = lambda x: x

        type_dim = 2 * scope.get('max_types', 1)

        x_in = keras.layers.Input((None, 3), name='rij')
        v_in = keras.layers.Input((None, type_dim), name='tij')
        w_in = None
        inputs = [x_in, v_in]
        if use_weights:
            w_in = keras.layers.Input((None,), name='wij')
            inputs = [x_in, v_in, w_in]

        dilation_dim = int(np.round(n_dim * dilation))

        def make_scorefun():
            layers = [keras.layers.Dense(dilation_dim)]

            layers.extend(normalization_getter('score'))

            layers.append(keras.layers.Activation(activation))
            if dropout:
                layers.append(keras.layers.Dropout(dropout))

            layers.append(keras.layers.Dense(1))
            return keras.models.Sequential(layers)

        def make_valuefun(dim, in_network=True):
            layers = []

            if in_network:
                layers.extend(normalization_getter('invariant_value'))

            layers.append(keras.layers.Dense(dilation_dim))
            layers.extend(normalization_getter('value'))

            layers.append(keras.layers.Activation(activation))
            if dropout:
                layers.append(keras.layers.Dropout(dropout))

            layers.append(keras.layers.Dense(dim))
            return keras.models.Sequential(layers)

        def make_block(last_x, last):
            residual_in_x = last_x
            residual_in = last
            if self.arguments['use_multivectors']:
                arg = [last_x, last, w_in] if use_weights else [last_x, last]
                last_x = gala.Multivector2MultivectorAttention(
                    make_scorefun(),
                    make_valuefun(n_dim),
                    make_valuefun(1),
                    False,
                    rank=rank,
                    join_fun=join_fun,
                    merge_fun=merge_fun,
                    invariant_mode=self.arguments['invar_mode'],
                )(arg)

            arg = [last_x, last, w_in] if use_weights else [last_x, last]
            last = Attention(
                make_scorefun(),
                make_valuefun(n_dim),
                False,
                rank=rank,
                join_fun=join_fun,
                merge_fun=merge_fun,
                invariant_mode=self.arguments['invar_mode'],
            )(arg)

            if block_nonlin:
                last = make_valuefun(n_dim, in_network=False)(last)

            if residual:
                last = last + residual_in

            for layer in normalization_getter('block'):
                last = layer(last)

            if self.arguments['use_multivectors']:
                last_x = residual_in_x + last_x
                for layer in normalization_getter('equivariant_value'):
                    last_x = layer(last_x)

            return last_x, last

        def make_vector_block(rs, vs):
            residual_in = rs
            rs = AttentionVector(
                make_scorefun(),
                make_valuefun(n_dim),
                make_valuefun(1),
                False,
                rank=rank,
                join_fun=join_fun,
                invariant_mode=invar_mode,
                covariant_mode=covar_mode,
                merge_fun=merge_fun,
            )([rs, vs])

            if residual:
                rs = rs + residual_in

            for layer in normalization_getter('equivariant_value'):
                rs = layer(rs)

            return rs

        last_x = x_in
        if distance_norm in ('mean', 'min'):
            last_x = NeighborDistanceNormalization(distance_norm)(last_x)
        elif distance_norm:
            raise NotImplementedError(distance_norm)

        last_x = maybe_upcast_vector(last_x)
        last = keras.layers.Dense(n_dim)(v_in)
        for _ in range(num_blocks):
            last_x, last = make_block(last_x, last)

        embedding = last

        for _ in range(self.arguments['num_vector_blocks']):
            last_x = make_vector_block(last_x, last)

        last_x = maybe_downcast_vector(last_x)

        embedding_model = keras.models.Model(inputs, embedding)

        scope['input_symbol'] = inputs
        scope['output'] = last_x
        scope['model'] = keras.models.Model(inputs, scope['output'])
        scope['embedding_model'] = embedding_model
