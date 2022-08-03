import collections

import flowws
from flowws import Argument as Arg
from tensorflow import keras


class LimitAccuracy(keras.callbacks.Callback):
    def __init__(self, threshold, batch_buffer=128, **kwargs):
        self.threshold = threshold
        self.batch_buffer = batch_buffer
        self._batches = collections.deque(maxlen=batch_buffer)
        super().__init__(**kwargs)

    def _check(self, value):
        if value >= self.threshold:
            self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        if self.batch_buffer:
            self._batches.append(logs['accuracy'])
            if len(self._batches) == self.batch_buffer:
                self._check(sum(self._batches) / self.batch_buffer)

    def on_epoch_end(self, epoch, logs=None):
        self._check(logs['accuracy'])

        if 'val_accuracy' in logs:
            self._check(logs['val_accuracy'])

    def get_config(self):
        result = super().get_config()
        result['threshold'] = self.threshold
        result['batch_buffer'] = self.batch_buffer
        return result


@flowws.add_stage_arguments
class LimitAccuracyCallback(flowws.Stage):

    ARGS = [
        Arg(
            'class_count_scale',
            '-c',
            float,
            1.0,
            help='Modification factor for number of classes',
        ),
    ]

    def run(self, scope, storage):
        N = scope['num_classes'] * self.arguments['class_count_scale']
        factor = 1.0 / N

        callback = LimitAccuracy(factor)
        scope.setdefault('callbacks', []).append(callback)
