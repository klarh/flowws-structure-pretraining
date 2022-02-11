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
