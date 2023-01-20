import tensorflow as tf
from tensorflow import keras


class GradientLayer(keras.layers.Layer):
    """Calculates the gradient of one input with respect to the other."""

    def call(self, inputs):
        return tf.gradients(inputs[0], inputs[1])


class NeighborDistanceNormalization(keras.layers.Layer):
    def __init__(self, mode='min', *args, **kwargs):
        self.mode = mode
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        if self.mode == 'min':
            distances = tf.linalg.norm(inputs, axis=-1, keepdims=True)
            scale = 1.0 / tf.maximum(
                1e-7, tf.math.reduce_min(distances, axis=-2, keepdims=True)
            )
        elif self.mode == 'mean':
            distances = tf.linalg.norm(inputs, axis=-1, keepdims=True)
            scale = 1.0 / tf.maximum(
                1e-7, tf.math.reduce_mean(distances, axis=-2, keepdims=True)
            )
        else:
            raise NotImplementedError()

        return inputs * scale

    def get_config(self):
        result = super().get_config()
        result['mode'] = self.mode
        return result


class NeighborhoodReduction(keras.layers.Layer):
    """Reduce values over the local neighborhood axis."""

    def __init__(self, mode='sum', *args, **kwargs):
        self.mode = mode

        super().__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        result = inputs
        if mask is not None:
            mask = tf.expand_dims(mask, -1)
            result = tf.where(mask, inputs, tf.zeros_like(inputs))

        if self.mode == 'sum':
            return tf.math.reduce_sum(result, axis=-2)
        elif self.mode == 'soft_max':
            return tf.math.reduce_logsumexp(result, axis=-2)
        else:
            raise NotImplementedError()

    def get_config(self):
        result = super().get_config()
        result['mode'] = self.mode
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

        # (..., N, 3) -> (..., N)
        mask = tf.reduce_all(tf.not_equal(inputs, 0), axis=-1)
        # (..., N, N)
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

        mask = tf.reduce_any(tf.not_equal(inputs, 0), axis=-1)
        mask = tf.logical_and(mask[..., None], mask[..., None, :])
        return mask


class SumLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.reduce_sum(inputs)
