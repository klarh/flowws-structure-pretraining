from .internal import NeighborDistanceNormalization

import flowws
from flowws import Argument as Arg
from geometric_algebra_attention import keras as gala
from tensorflow import keras
import numpy as np
import tensorflow as tf

NORMALIZATION_LAYERS = {
    None: lambda _: [],
    'none': lambda _: [],
    'batch': lambda _: [keras.layers.BatchNormalization()],
    'layer': lambda _: [keras.layers.LayerNormalization()],
    'momentum': lambda _: [gala.MomentumNormalization()],
    'momentum_layer': lambda _: [gala.MomentumLayerNormalization()],
}

NORMALIZATION_LAYER_DOC = ' (any of {})'.format(
    ','.join(map(str, NORMALIZATION_LAYERS))
)


class VAESampler(keras.layers.Layer):
    def __init__(self, dim, loss_sum_axes=(-1,), loss_scale=1.0, *args, **kwargs):
        self.dim = dim
        self.loss_sum_axes = loss_sum_axes
        self.loss_scale = loss_scale

        self.z_mean_projection = keras.layers.Dense(self.dim)
        self.z_log_var_projection = keras.layers.Dense(self.dim)

        super().__init__(*args, **kwargs)

    def call(self, inputs):
        z_mean = self.z_mean_projection(inputs)
        z_log_var = self.z_log_var_projection(inputs)
        shape = tf.shape(z_mean)
        loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        loss = tf.reduce_mean(tf.reduce_sum(loss, self.loss_sum_axes))
        loss = loss * self.loss_scale
        self.add_metric(loss, name='kl_loss')
        self.add_loss(loss)
        return tf.random.normal(shape, z_mean, tf.exp(0.5 * z_log_var))

    def get_config(self):
        result = super().get_config()
        result['dim'] = self.dim
        result['loss_sum_axes'] = tuple(self.loss_sum_axes)
        result['loss_scale'] = self.loss_scale
        return result


def expand_neighborhood_dim(x):
    return x[..., None, :]


def stack_vector_layers(
    AttentionVector,
    r,
    v,
    n_vectors,
    scorefun,
    valuefun,
    scalefun,
    join_fun,
    merge_fun,
    n_dim=32,
    rank=2,
    invar_mode='full',
    covar_mode='full',
    include_normalized_products=False,
):
    pieces = []
    for _ in range(n_vectors):
        layer = AttentionVector(
            scorefun(),
            valuefun(n_dim),
            scalefun(1),
            reduce=True,
            invariant_mode=invar_mode,
            covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            merge_fun=merge_fun,
            join_fun=join_fun,
            rank=rank,
        )
        piece = layer([r, v])
        piece = keras.layers.Lambda(expand_neighborhood_dim)(piece)
        pieces.append(piece)

    return keras.layers.Concatenate(axis=-2)(pieces)


class SVDLayer(keras.layers.Layer):
    def build(self, input_shape):
        self.num_vectors, self.n_dim = input_shape[-2:]
        return super().build(input_shape)

    def call(self, inputs):
        s, u, v = tf.linalg.svd(inputs)
        result = tf.matmul(u, v, adjoint_b=True)
        if self.num_vectors < self.n_dim:
            cross = tf.linalg.cross(result[..., 0, :], result[..., 1, :])[..., None, :]
            result = tf.concat([result, cross], axis=-2)
        return result


class NormalizeLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.linalg.normalize(inputs, axis=-1)[0]


class StaticEmbedding(keras.layers.Embedding):
    def call(self, inputs):
        return super().call(tf.range(0, self.input_dim)[None, :])


@flowws.add_stage_arguments
class GalaBottleneckAutoencoder(flowws.Stage):
    """Reproduce point clouds after a bottleneck layer"""

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
            'layer',
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
        Arg('variational', '-v', bool, True, help='If True, make a VAE'),
        Arg('vae_dim', None, int, 8, help='Dimensionality of latent space for VAE'),
        Arg(
            'vae_scale',
            None,
            float,
            1e-5,
            help='Loss term scale for variational component',
        ),
        Arg(
            'num_reference_vectors',
            None,
            int,
            2,
            help='Number of reference vectors to produce',
        ),
        Arg(
            'use_multivectors',
            None,
            bool,
            False,
            help='If True, use multivector intermediates for calculations',
        ),
        Arg(
            'cross_attention',
            None,
            bool,
            True,
            help='If True, generate embeddings using cross-attention between the '
            'generated basis set and vector intermediates',
        ),
        Arg(
            'include_normalized_products',
            None,
            bool,
            False,
            help='Also include normalized geometric product terms',
        ),
        Arg(
            'normalize_equivariant_values',
            None,
            bool,
            False,
            help='If True, multiply vector values by normalized vectors at each attention step',
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
        n_ref = self.arguments['num_reference_vectors']
        DropoutLayer = scope.get('dropout_class', keras.layers.Dropout)

        normalization_getter = lambda key: (
            NORMALIZATION_LAYERS[self.arguments.get(key + '_normalization', None)](rank)
        )

        if self.arguments['use_multivectors']:
            Attention = gala.MultivectorAttention
            AttentionVector = gala.Multivector2MultivectorAttention
            AttentionLabeled = gala.LabeledMultivectorAttention
            maybe_upcast_vector = gala.Vector2Multivector()
            maybe_downcast_vector = gala.Multivector2Vector()
        else:
            Attention = gala.VectorAttention
            AttentionVector = gala.Vector2VectorAttention
            AttentionLabeled = gala.LabeledVectorAttention
            maybe_upcast_vector = lambda x: x
            maybe_downcast_vector = lambda x: x

        type_dim = 2 * scope.get('max_types', 1)
        dilation_dim = int(np.round(n_dim * dilation))

        def make_layer_inputs(x, v):
            nonnorm = (x, v, w_in) if use_weights else (x, v)
            if self.arguments['normalize_equivariant_values']:
                xnorm = keras.layers.LayerNormalization()(x)
                norm = (xnorm, v, w_in) if use_weights else (xnorm, v)
                return [nonnorm] + (rank - 1) * [norm]
            else:
                return rank * [nonnorm]

        def make_scorefun():
            layers = [keras.layers.Dense(dilation_dim)]

            layers.extend(normalization_getter('score'))

            layers.append(keras.layers.Activation(activation))
            if dropout:
                layers.append(DropoutLayer(dropout))

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
                layers.append(DropoutLayer(dropout))

            layers.append(keras.layers.Dense(dim))
            return keras.models.Sequential(layers)

        def make_block(last_x, last):
            residual_in_x = last_x
            residual_in = last
            if self.arguments['use_multivectors']:
                arg = make_layer_inputs(last_x, last)
                last_x = gala.Multivector2MultivectorAttention(
                    make_scorefun(),
                    make_valuefun(n_dim),
                    make_valuefun(1),
                    False,
                    rank=rank,
                    join_fun=join_fun,
                    merge_fun=merge_fun,
                    invariant_mode=invar_mode,
                    covariant_mode=covar_mode,
                    include_normalized_products=self.arguments[
                        'include_normalized_products'
                    ],
                )(arg)

            arg = make_layer_inputs(last_x, last)
            last = Attention(
                make_scorefun(),
                make_valuefun(n_dim),
                False,
                rank=rank,
                join_fun=join_fun,
                merge_fun=merge_fun,
                invariant_mode=invar_mode,
                covariant_mode=covar_mode,
                include_normalized_products=self.arguments[
                    'include_normalized_products'
                ],
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
                include_normalized_products=self.arguments[
                    'include_normalized_products'
                ],
                merge_fun=merge_fun,
            )([rs, vs])

            if residual:
                rs = rs + residual_in

            for layer in normalization_getter('equivariant_value'):
                rs = layer(rs)

            return rs

        def make_labeled_block(last):
            last_x = AttentionLabeled(
                make_scorefun(),
                make_valuefun(n_dim),
                make_valuefun(1),
                True,
                rank=rank,
                join_fun=join_fun,
                merge_fun=merge_fun,
                invariant_mode=invar_mode,
                covariant_mode=covar_mode,
                include_normalized_products=self.arguments[
                    'include_normalized_products'
                ],
            )([last, (reference_last_x, reference_embedding)])

            return last_x

        if 'encoded_base' in scope:
            (last_x, last) = scope['encoded_base']
            inputs = scope['input_symbol']

            if self.arguments['transfer_freeze']:
                frozen_model = keras.models.Model(inputs, scope['encoded_base'])
                frozen_model.trainable = False
                (last_x, last) = frozen_model(inputs)
        else:
            x_in = keras.layers.Input((None, 3), name='rij')
            v_in = keras.layers.Input((None, type_dim), name='tij')
            w_in = None
            inputs = [x_in, v_in]
            if use_weights:
                w_in = keras.layers.Input((None,), name='wij')
                inputs = [x_in, v_in, w_in]

            last_x = x_in
            if distance_norm in ('mean', 'min'):
                last_x = NeighborDistanceNormalization(distance_norm)(last_x)
            elif distance_norm == 'none':
                pass
            elif distance_norm:
                raise NotImplementedError(distance_norm)

            last_x = maybe_upcast_vector(last_x)
            last = keras.layers.Dense(n_dim)(v_in)
            for _ in range(num_blocks):
                last_x, last = make_block(last_x, last)

            reference_last_x = stack_vector_layers(
                AttentionVector,
                last_x,
                last,
                n_ref,
                make_scorefun,
                make_valuefun,
                make_valuefun,
                join_fun,
                merge_fun,
                n_dim,
                rank,
                invar_mode,
                covar_mode,
                include_normalized_products=self.arguments[
                    'include_normalized_products'
                ],
            )

            reference_last_x = SVDLayer()(maybe_downcast_vector(reference_last_x))
            reference_last_x = maybe_upcast_vector(reference_last_x)
            n_ref = max(n_ref, 3)

            reference_embedding = StaticEmbedding(n_ref, n_dim)(x_in)

            embedding = last
            if self.arguments['cross_attention']:
                arg = [last_x, last, w_in] if use_weights else [last_x, last]
                arg = [arg, [reference_last_x, reference_embedding]]
                embedding = last = Attention(
                    make_scorefun(),
                    make_valuefun(n_dim),
                    False,
                    rank=rank,
                    join_fun=join_fun,
                    merge_fun=merge_fun,
                    invariant_mode=invar_mode,
                    covariant_mode=covar_mode,
                    include_normalized_products=self.arguments[
                        'include_normalized_products'
                    ],
                )(arg)

            if self.arguments['variational']:
                samp = VAESampler(
                    self.arguments['vae_dim'], (-1,), self.arguments['vae_scale']
                )
                embedding = samp.z_mean_projection(last)
                last = samp(last)
                if n_dim != self.arguments['vae_dim']:
                    last = keras.layers.Dense(n_dim)(last)

            last_x = make_labeled_block(last)

            scope['encoded_base'] = (last_x, last)
            embedding_model = keras.models.Model(inputs, embedding)
            scope['embedding_model'] = embedding_model

        for _ in range(self.arguments['num_vector_blocks']):
            last_x = make_vector_block(last_x, last)

        last_x = maybe_downcast_vector(last_x)

        scope['input_symbol'] = inputs
        scope['output'] = last_x
        scope['model'] = keras.models.Model(inputs, scope['output'])
