import tensorflow as tf
from tensorflow import keras


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
