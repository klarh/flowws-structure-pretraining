import logging

import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


@tf.custom_gradient
def custom_norm(x):
    """Calculate the norm of a set of vector-like quantities, with some
    numeric stabilization applied to the gradient."""
    y = tf.linalg.norm(x, axis=-1, keepdims=True)

    def grad(dy):
        y = custom_norm(x)
        return dy * (x / tf.maximum(y, 1e-19))

    return y, grad


class GradientLayer(keras.layers.Layer):
    """Calculates the gradient of one input with respect to the other."""

    def call(self, inputs):
        result = tf.gradients(inputs[0], inputs[1])
        if result[0] is None:
            msg = (
                'Shortcutting gradient calculation in GradientLayer; '
                'this can happen while building the model, but should not '
                'happen during training'
            )
            logger.warning(msg)
            return tf.zeros_like(inputs[1])
        return result


class NeighborDistanceNormalization(keras.layers.Layer):
    def __init__(
        self, mode='min', lengthscale=1.0, return_scale=False, *args, **kwargs
    ):
        self.mode = mode
        self.lengthscale = float(lengthscale)
        self.return_scale = return_scale

        if mode == 'mean':
            self.reduction = NeighborhoodReduction(mode)
        super().__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        if self.mode == 'min':
            distances = custom_norm(inputs)
            scale = self.lengthscale / tf.maximum(
                1e-7, tf.math.reduce_min(distances, axis=-2, keepdims=True)
            )
        elif self.mode == 'min_nonzero':
            distances = custom_norm(inputs)
            filtered = tf.where(distances == 0.0, 1e9, distances)
            scale = self.lengthscale / tf.maximum(
                1e-7, tf.math.reduce_min(filtered, axis=-2, keepdims=True)
            )
        elif self.mode == 'mean':
            distances = custom_norm(inputs)
            denominator = self.reduction(distances, mask=mask)
            denominator_inverse = tf.math.reciprocal_no_nan(denominator)
            scale = self.lengthscale * denominator_inverse
        else:
            raise NotImplementedError()

        if self.return_scale:
            return inputs * scale, scale
        return inputs * scale

    def compute_mask(self, inputs, mask=None):
        if self.return_scale:
            return mask, mask
        return mask

    def get_config(self):
        result = super().get_config()
        result['lengthscale'] = self.lengthscale
        result['mode'] = self.mode
        result['return_scale'] = self.return_scale
        return result


class NeighborhoodReduction(keras.layers.Layer):
    """Reduce values over the local neighborhood axis."""

    def __init__(self, mode='sum', keepdims=True, *args, **kwargs):
        self.mode = mode
        self.keepdims = keepdims

        super().__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        result = inputs
        if mask is not None:
            mask = tf.expand_dims(mask, -1)
            result = tf.where(mask, inputs, tf.zeros_like(inputs))

        if self.mode == 'sum':
            return tf.math.reduce_sum(result, axis=-2, keepdims=self.keepdims)
        elif self.mode == 'soft_max':
            return tf.math.reduce_logsumexp(result, axis=-2, keepdims=self.keepdims)
        elif self.mode == 'mean':
            numerator = tf.math.reduce_sum(result, axis=-2, keepdims=self.keepdims)
            if mask is not None:
                denominator = tf.math.reduce_sum(
                    tf.cast(mask, tf.float32), axis=-2, keepdims=self.keepdims
                )
                denominator = tf.cast(denominator, tf.float32)
                denominator_inverse = tf.math.reciprocal_no_nan(denominator)
            else:
                denominator_inverse = 1.0 / tf.cast(tf.shape(inputs)[-2], tf.float32)
            return numerator * denominator_inverse
        else:
            raise NotImplementedError()

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask

        return tf.math.reduce_any(mask, axis=-1, keepdims=self.keepdims)

    def get_config(self):
        result = super().get_config()
        result['mode'] = self.mode
        result['keepdims'] = self.keepdims
        return result


class NoiseInjector(keras.layers.Layer):
    def __init__(self, magnitude, only_during_training=True, **kwargs):
        self.magnitude = magnitude
        self.only_during_training = only_during_training
        super().__init__(**kwargs)

    def call(self, inputs, training=False):
        if self.only_during_training and not training:
            return inputs

        noise = tf.random.normal(tf.shape(inputs), stddev=self.magnitude)
        return inputs + noise

    def get_config(self):
        result = super().get_config()
        result['magnitude'] = self.magnitude
        result['only_during_training'] = self.only_during_training
        return result


class PairwiseVectorDifference(keras.layers.Layer):
    """Calculate the difference of all pairs of vectors in the neighborhood axis."""

    def call(self, inputs):
        return inputs[..., None, :] - inputs[..., None, :, :]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask

        # (..., N) -> (..., N, N)
        mask = tf.logical_and(mask[..., None], mask[..., None, :])
        return mask


class PairwiseVectorDifferenceSum(keras.layers.Layer):
    """Calculate the symmetric difference and sum of all pairs of vectors in the neighborhood axis."""

    def call(self, inputs):
        return tf.concat(
            [
                inputs[..., None, :] - inputs[..., None, :, :],
                inputs[..., None, :] + inputs[..., None, :, :],
            ],
            axis=-1,
        )

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask

        # (..., N) -> (..., N, N)
        mask = tf.logical_and(mask[..., None], mask[..., None, :])
        return mask


class ResidualMaskedLayer(keras.layers.Layer):
    def call(self, inputs):
        left, right = inputs
        return left + right

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = (None, None)
        (mask_left, mask_right) = mask
        if mask_left is None:
            return mask_right
        elif mask_right is None:
            return mask_left
        return tf.math.logical_and(mask_left, mask_right)


class LambdaMaskedLayer(keras.layers.Lambda):
    def compute_mask(self, inputs, mask=None):
        return mask


class ScaledMSELoss(keras.losses.MeanSquaredError):
    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, y_true, y_pred):
        return self.beta * super().call(y_true, y_pred)

    def get_config(self):
        result = super().get_config()
        result['beta'] = self.beta
        return result


class SumLayer(keras.layers.Layer):
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.where(mask[..., None], inputs, tf.zeros_like(inputs))
        return tf.math.reduce_sum(inputs)


class ZeroMaskingLayer(keras.layers.Layer):
    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.not_equal(inputs, 0), axis=-1)
        return mask


class IdentityLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return inputs
