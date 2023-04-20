from .GalaCore import GalaCore

import flowws
from flowws import Argument as Arg
from tensorflow import keras


@flowws.add_stage_arguments
class GalaBondClassifier(GalaCore):
    """Classify bonds using geometric algebra attention"""

    ARGS = GalaCore.ARGS + [
        Arg(
            'l1_activity_regularization',
            None,
            float,
            help='L1 activity regularization of outputs',
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
        num_classes = scope.get('num_classes', 2)

        self._init(scope, storage)
        if 'encoded_base' not in scope:
            super().run(scope, storage)

        (last_x, last) = scope['encoded_base']
        inputs = scope['input_symbol']

        if self.arguments['transfer_freeze']:
            frozen_model = keras.models.Model(inputs, scope['encoded_base'])
            frozen_model.trainable = False
            (last_x, last) = frozen_model(inputs)

        arg = self.make_layer_inputs(last_x, last)
        (last, ivs, att) = self.Attention(
            self.make_scorefun(),
            self.make_valuefun(self.n_dim),
            scope.get('per_cloud', False),
            name='final_attention',
            rank=self.rank,
            join_fun=self.join_fun,
            merge_fun=self.merge_fun,
            invariant_mode=self.invar_mode,
            covariant_mode=self.covar_mode,
            include_normalized_products=self.arguments['include_normalized_products'],
        )(arg, return_invariants=True, return_attention=True)

        embedding = last

        if 'embedding_dimension' in self.arguments:
            last = keras.layers.Dense(self.arguments['embedding_dimension'])(last)
            embedding = last

        if scope.get('multilabel', False):
            if scope.get('multilabel_softmax', False):
                # softmax + categorical crossentropy
                last = keras.layers.Dense(2 * num_classes)(last)
                last = keras.layers.Reshape((num_classes, 2))(last)
                last = keras.layers.Activation('softmax')(last)
            else:
                # sigmoid + binary crossentropy
                last = keras.layers.Dense(num_classes, activation='sigmoid')(last)
        else:
            # softmax + categorical crossentropy
            last = keras.layers.Dense(num_classes, activation='softmax')(last)

        if 'l1_activity_regularization' in self.arguments:
            last = keras.layers.ActivityRegularization(
                l1=self.arguments['l1_activity_regularization']
            )(last)

        scope['input_symbol'] = inputs
        scope['output'] = last
        scope['model'] = keras.models.Model(inputs, scope['output'])
        scope['embedding_model'] = keras.models.Model(inputs, embedding)
