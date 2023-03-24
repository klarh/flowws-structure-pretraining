from .GalaCore import GalaCore

import flowws
from flowws import Argument as Arg
from geometric_algebra_attention import keras as gala
from tensorflow import keras
import tensorflow as tf


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
    convex_covariants=False,
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
            convex_covariants=convex_covariants,
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
class GalaBottleneckAutoencoder(GalaCore):
    """Reproduce point clouds after a bottleneck layer"""

    ARGS = GalaCore.ARGS + [
        Arg(
            'num_vector_blocks',
            None,
            int,
            1,
            help='Number of vector-valued blocks to use',
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
            'cross_attention',
            None,
            bool,
            True,
            help='If True, generate embeddings using cross-attention between the '
            'generated basis set and vector intermediates',
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
        n_ref = self.arguments['num_reference_vectors']

        def make_labeled_block(last):
            last_x = self.AttentionLabeled(
                self.make_scorefun(),
                self.make_valuefun(self.n_dim),
                self.make_valuefun(1),
                True,
                rank=self.rank,
                join_fun=self.join_fun,
                merge_fun=self.merge_fun,
                invariant_mode=self.invar_mode,
                covariant_mode=self.covar_mode,
                include_normalized_products=self.arguments[
                    'include_normalized_products'
                ],
                convex_covariants=self.arguments['convex_covariants'],
            )([last, (reference_last_x, reference_embedding)])

            return last_x

        self._init(scope, storage)
        if 'encoded_base' in scope:
            (last_x, last) = scope['encoded_base']
            inputs = scope['input_symbol']

            if self.arguments['transfer_freeze']:
                frozen_model = keras.models.Model(inputs, scope['encoded_base'])
                frozen_model.trainable = False
                (last_x, last) = frozen_model(inputs)
        else:
            super().run(scope, storage)

            (last_x, last) = scope['encoded_base']
            inputs = scope['input_symbol']

            if self.arguments['transfer_freeze']:
                frozen_model = keras.models.Model(inputs, scope['encoded_base'])
                frozen_model.trainable = False
                (last_x, last) = frozen_model(inputs)

            reference_last_x = stack_vector_layers(
                self.AttentionVector,
                last_x,
                last,
                n_ref,
                self.make_scorefun,
                self.make_valuefun,
                self.make_valuefun,
                self.join_fun,
                self.merge_fun,
                self.n_dim,
                self.rank,
                self.invar_mode,
                self.covar_mode,
                include_normalized_products=self.arguments[
                    'include_normalized_products'
                ],
                convex_covariants=self.arguments['convex_covariants'],
            )

            reference_last_x = SVDLayer()(self.maybe_downcast_vector(reference_last_x))
            reference_last_x = self.maybe_upcast_vector(reference_last_x)
            n_ref = max(n_ref, 3)

            reference_embedding = StaticEmbedding(n_ref, self.n_dim)(inputs[0])

            embedding = last
            if self.arguments['cross_attention']:
                arg = [last_x, last, w_in] if self.use_weights else [last_x, last]
                arg = [arg, [reference_last_x, reference_embedding]]
                embedding = last = self.Attention(
                    self.make_scorefun(),
                    self.make_valuefun(self.n_dim),
                    False,
                    rank=self.rank,
                    join_fun=self.join_fun,
                    merge_fun=self.merge_fun,
                    invariant_mode=self.invar_mode,
                    covariant_mode=self.covar_mode,
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
                if self.n_dim != self.arguments['vae_dim']:
                    last = keras.layers.Dense(self.n_dim)(last)

            last_x = make_labeled_block(last)

            scope['encoded_base'] = (last_x, last)
            embedding_model = keras.models.Model(inputs, embedding)
            scope['embedding_model'] = embedding_model

        for _ in range(self.arguments['num_vector_blocks']):
            last_x = self.make_vector_block(last_x, last)

        last_x = self.maybe_downcast_vector(last_x)

        if 'equivariant_rescale_factor' in scope:
            last_x = last_x / scope['equivariant_rescale_factor']

        scope['input_symbol'] = inputs
        scope['output'] = last_x
        scope['model'] = keras.models.Model(inputs, scope['output'])
