from .internal import NeighborDistanceNormalization, NoiseInjector
from .internal import PairwiseVectorDifference, PairwiseVectorDifferenceSum
from .internal import ResidualMaskedLayer, ZeroMaskingLayer

import flowws
from flowws import Argument as Arg
from geometric_algebra_attention import keras as gala
import numpy as np
import tensorflow as tf
from tensorflow import keras


LAMBDA_ACTIVATIONS = {
    'gaussian': lambda x: tf.math.exp(-tf.square(x)),
    'leakyswish': lambda x: tf.nn.swish(x) - 1e-2 * tf.nn.swish(-x),
    'log1pswish': lambda x: tf.math.log1p(tf.nn.swish(x)),
    'sin': tf.sin,
}

NORMALIZATION_LAYERS = {
    None: lambda _, **kwargs: [],
    'none': lambda _, **kwargs: [],
    'batch': lambda _, **kwargs: [keras.layers.BatchNormalization()],
    'layer': lambda _, **kwargs: [keras.layers.LayerNormalization()],
    'momentum': lambda _, **kwargs: [
        gala.MomentumNormalization(momentum=kwargs.get('momentum', 0.99))
    ],
    'momentum_layer': lambda _, **kwargs: [
        gala.MomentumLayerNormalization(momentum=kwargs.get('momentum', 0.99))
    ],
}

NORMALIZATION_LAYER_DOC = ' (any of {})'.format(
    ','.join(map(str, NORMALIZATION_LAYERS))
)


@flowws.add_stage_arguments
class GalaCore(flowws.Stage):
    """Basic task-agnostic core architecture using algebra attention"""

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
            'mean',
            help='Function to use for joining invariant and node-level signals',
        ),
        Arg(
            'merge_fun',
            '-m',
            str,
            'mean',
            help='Function to use for merging node-level signals',
        ),
        Arg('dropout', '-d', float, 0.5, help='Dropout probability within network'),
        Arg('num_blocks', None, int, 1, help='Number of blocks to use'),
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
            'mean',
            help='Create scale-invariant networks by normalizing neighbor distances (mean/min)',
        ),
        Arg('invar_mode', '-i', str, 'full', help='Rotation-invariant mode switch'),
        Arg('covar_mode', '-c', str, 'full', help='Rotation-equivariant mode switch'),
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
        Arg(
            'use_multivectors',
            None,
            bool,
            False,
            help='If True, use multivector intermediates for calculations',
        ),
        Arg(
            'embedding_dimension',
            None,
            int,
            help='Dimension to use for pre-classification projection embedding',
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
            'inject_noise',
            None,
            float,
            0,
            help='If given, add random noise with the given magnitude to input coordinates',
        ),
        Arg(
            'normalization_kwargs',
            None,
            [(str, eval)],
            [],
            help='Keyword arguments to pass to normalization functions',
        ),
        Arg(
            'linear_invariant_projection',
            None,
            bool,
            False,
            help='If True, use a simple linear projection for value nets rather than an MLP',
        ),
        Arg(
            'scale_equivariant_factor',
            None,
            float,
            help='If given, normalize input vectors to have length around the given factor',
        ),
        Arg(
            'scale_equivariant_mode',
            None,
            str,
            'mean',
            help='Summary statistic to use to normalize vectors if scale_equivariant_factor is enabled',
        ),
        Arg(
            'scale_equivariant_embedding',
            None,
            bool,
            False,
            help='Use a learned rotation-invariant embedding for the scale-equivariant factor',
        ),
        Arg(
            'direct_scale_equivariant_embedding',
            None,
            bool,
            False,
            help='Use non-reciprocal distance embedding for the scale-equivariant factor',
        ),
        Arg(
            'gaussian_scale_equivariant_embedding',
            None,
            bool,
            False,
            help='Use Gaussian distance embedding for the scale-equivariant factor',
        ),
        Arg(
            'convex_covariants',
            None,
            bool,
            False,
            help='If True, use convex combinations of covariant values',
        ),
        Arg('mlp_layers', None, int, 1, help='Number of hidden layers to use in MLPs'),
        Arg(
            'tied_attention',
            None,
            bool,
            False,
            help='If True, use tied attention weights',
        ),
    ]

    def _init(self, scope, storage):
        self.use_weights = scope.get('use_bond_weights', False)
        self.n_dim = self.arguments['n_dim']
        dilation = self.arguments['dilation_factor']
        self.dilation_dim = int(np.round(self.n_dim * dilation))
        self.block_nonlin = self.arguments['block_nonlinearity']
        self.residual = self.arguments['residual']
        self.join_fun = self.arguments['join_fun']
        self.merge_fun = self.arguments['merge_fun']
        self.dropout = self.arguments['dropout']
        self.rank = self.arguments['rank']
        self.invar_mode = self.arguments['invar_mode']
        self.covar_mode = self.arguments['covar_mode']
        self.DropoutLayer = scope.get('dropout_class', keras.layers.Dropout)

        normalization_kwargs = dict(self.arguments.get('normalization_kwargs', []))
        self.normalization_getter = lambda key: (
            NORMALIZATION_LAYERS[self.arguments.get(key + '_normalization', None)](
                self.rank, **normalization_kwargs
            )
        )

        if self.arguments['use_multivectors']:
            self.Attention = gala.MultivectorAttention
            self.AttentionVector = gala.Multivector2MultivectorAttention
            self.AttentionLabeled = gala.LabeledMultivectorAttention
            self.AttentionTied = gala.TiedMultivectorAttention
            self.maybe_upcast_vector = gala.Vector2Multivector()
            self.maybe_downcast_vector = gala.Multivector2Vector()
        else:
            self.Attention = gala.VectorAttention
            self.AttentionVector = gala.Vector2VectorAttention
            self.AttentionLabeled = gala.LabeledVectorAttention
            self.AttentionTied = gala.TiedVectorAttention
            self.maybe_upcast_vector = lambda x: x
            self.maybe_downcast_vector = lambda x: x

        if self.arguments['activation'] in LAMBDA_ACTIVATIONS:
            self.activation_layer = lambda: keras.layers.Lambda(
                LAMBDA_ACTIVATIONS[self.arguments['activation']]
            )
        else:
            self.activation_layer = lambda: keras.layers.Activation(
                self.arguments['activation']
            )

    def run(self, scope, storage):
        self._init(scope, storage)
        distance_norm = self.arguments['normalize_distances']
        num_blocks = self.arguments['num_blocks']

        type_dim = scope.get('max_types', 1)
        type_dim *= 1 if scope.get('per_molecule', False) else 2
        type_dim = scope.get('type_embedding_size', type_dim)

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
            if self.use_weights:
                w_in = keras.layers.Input((None,), name='wij')
                inputs = [x_in, v_in, w_in]

            (last_x, last) = self.maybe_expand_molecule(scope, x_in, v_in)
            last_x = ZeroMaskingLayer()(last_x)
            last = keras.layers.Dense(self.n_dim, name='type_embedding')(last)

            scope.pop('equivariant_rescale_factor', None)
            if self.arguments.get('scale_equivariant_factor', None):
                scale = self.arguments['scale_equivariant_factor']
                norm_mode = self.arguments['scale_equivariant_mode']
                last_x, rescale = NeighborDistanceNormalization(norm_mode, scale, True)(
                    last_x
                )
                scope['equivariant_rescale_factor'] = rescale[..., None]

                if self.arguments.get('scale_equivariant_embedding', None):
                    embedding_scale = rescale
                    if (
                        self.arguments['direct_scale_equivariant_embedding']
                        or self.arguments['gaussian_scale_equivariant_embedding']
                    ):
                        embedding_scale = tf.math.reciprocal(embedding_scale)
                    embedding = keras.layers.Dense(
                        self.n_dim, name='distance_embedding'
                    )(embedding_scale)
                    if self.arguments['gaussian_scale_equivariant_embedding']:
                        embedding = keras.layers.Lambda(LAMBDA_ACTIVATIONS['gaussian'])(
                            embedding
                        )
                    last = last + embedding
            elif distance_norm in ('mean', 'min'):
                last_x = NeighborDistanceNormalization(distance_norm)(last_x)
            elif distance_norm == 'none':
                pass
            elif distance_norm:
                raise NotImplementedError(distance_norm)

            if self.arguments['inject_noise']:
                last_x = NoiseInjector(self.arguments['inject_noise'])(last_x)

            last_x = self.maybe_upcast_vector(last_x)
            for _ in range(num_blocks):
                last_x, last = self.make_block(last_x, last)

            scope['encoded_base'] = (last_x, last)

        embedding = last
        scope['input_symbol'] = inputs
        scope['output'] = (last_x, last)
        scope['model'] = keras.models.Model(inputs, scope['output'])
        scope['embedding_model'] = keras.models.Model(inputs, embedding)

    def maybe_expand_molecule(self, scope, last_x, last):
        if scope.get('per_molecule', False):
            last_x = PairwiseVectorDifference()(last_x)
            last = PairwiseVectorDifferenceSum()(last)
        return last_x, last

    def make_layer_inputs(self, x, v):
        nonnorm = (x, v, w_in) if self.use_weights else (x, v)
        if self.arguments['normalize_equivariant_values']:
            xnorm = keras.layers.LayerNormalization()(x)
            norm = (xnorm, v, w_in) if self.use_weights else (xnorm, v)
            return [nonnorm] + (self.rank - 1) * [norm]
        else:
            return self.rank * [nonnorm]

    def make_scorefun(self):
        layers = []

        for _ in range(self.arguments['mlp_layers']):
            layers.append(keras.layers.Dense(self.dilation_dim))

            layers.extend(self.normalization_getter('score'))

            layers.append(self.activation_layer())
            if self.dropout:
                layers.append(self.DropoutLayer(self.dropout))

        layers.append(keras.layers.Dense(1))
        return keras.models.Sequential(layers)

    def make_valuefun(self, dim, in_network=True):
        layers = []

        if in_network:
            layers.extend(self.normalization_getter('invariant_value'))

            if self.arguments['linear_invariant_projection']:
                layers.append(keras.layers.Dense(dim))
                return keras.models.Sequential(layers)

        for _ in range(self.arguments['mlp_layers']):
            layers.append(keras.layers.Dense(self.dilation_dim))
            layers.extend(self.normalization_getter('value'))

            layers.append(self.activation_layer())
            if self.dropout:
                layers.append(self.DropoutLayer(self.dropout))

        layers.append(keras.layers.Dense(dim))
        return keras.models.Sequential(layers)

    def make_block(self, last_x, last):
        residual_in_x = last_x
        residual_in = last

        if self.arguments['tied_attention']:
            arg = self.make_layer_inputs(last_x, last)
            (last_x, last) = self.AttentionTied(
                self.make_scorefun(),
                self.make_valuefun(self.n_dim),
                self.make_valuefun(1),
                False,
                rank=self.rank,
                join_fun=self.join_fun,
                merge_fun=self.merge_fun,
                invariant_mode=self.invar_mode,
                covariant_mode=self.covar_mode,
                include_normalized_products=self.arguments[
                    'include_normalized_products'
                ],
                convex_covariants=self.arguments['convex_covariants'],
            )(arg)
        else:
            if self.arguments['use_multivectors']:
                arg = self.make_layer_inputs(last_x, last)
                last_x = self.AttentionVector(
                    self.make_scorefun(),
                    self.make_valuefun(self.n_dim),
                    self.make_valuefun(1),
                    False,
                    rank=self.rank,
                    join_fun=self.join_fun,
                    merge_fun=self.merge_fun,
                    invariant_mode=self.invar_mode,
                    covariant_mode=self.covar_mode,
                    include_normalized_products=self.arguments[
                        'include_normalized_products'
                    ],
                    convex_covariants=self.arguments['convex_covariants'],
                )(arg)

            arg = self.make_layer_inputs(last_x, last)
            last = self.Attention(
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

        if self.block_nonlin:
            last = self.make_valuefun(self.n_dim, in_network=False)(last)

        if self.residual:
            last = ResidualMaskedLayer()((last, residual_in))

        for layer in self.normalization_getter('block'):
            last = layer(last)

        if self.arguments['use_multivectors']:
            last_x = ResidualMaskedLayer()((residual_in_x, last_x))
            for layer in self.normalization_getter('equivariant_value'):
                last_x = layer(last_x)

        return last_x, last

    def make_vector_block(self, rs, vs):
        residual_in = rs
        rs = self.AttentionVector(
            self.make_scorefun(),
            self.make_valuefun(self.n_dim),
            self.make_valuefun(1),
            False,
            rank=self.rank,
            join_fun=self.join_fun,
            merge_fun=self.merge_fun,
            invariant_mode=self.invar_mode,
            covariant_mode=self.covar_mode,
            include_normalized_products=self.arguments['include_normalized_products'],
            convex_covariants=self.arguments['convex_covariants'],
        )([rs, vs])

        if self.residual:
            rs = rs + residual_in

        for layer in self.normalization_getter('equivariant_value'):
            rs = layer(rs)

        return rs
