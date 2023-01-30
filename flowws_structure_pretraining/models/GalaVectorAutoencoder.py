from .GalaCore import GalaCore

import flowws
from flowws import Argument as Arg
from tensorflow import keras


@flowws.add_stage_arguments
class GalaVectorAutoencoder(GalaCore):
    """Reproduce input point clouds"""

    ARGS = GalaCore.ARGS + [
        Arg(
            'num_vector_blocks',
            None,
            int,
            1,
            help='Number of vector-valued blocks to use',
        ),
        Arg(
            'drop_geometric_embeddings',
            None,
            bool,
            False,
            help='If True, use vector inputs rather than geometric accumulations around the embedding layer',
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
        self._init(scope, storage)
        if 'encoded_base' not in scope:
            super().run(scope, storage)

        (last_x, last) = scope['encoded_base']
        inputs = scope['input_symbol']

        if self.arguments['transfer_freeze']:
            frozen_model = keras.models.Model(inputs, scope['encoded_base'])
            frozen_model.trainable = False
            (last_x, last) = frozen_model(inputs)

        if self.arguments['drop_geometric_embeddings']:
            arg = [last_x, last]
            arg[0] = self.maybe_upcast_vector(scope['input_symbol'][0])
        else:
            arg = self.make_layer_inputs(last_x, last)

        for _ in range(self.arguments['num_vector_blocks']):
            last_x = self.make_vector_block(last_x, last)

        last_x = self.maybe_downcast_vector(last_x)

        if 'equivariant_rescale_factor' in scope:
            last_x = last_x / scope['equivariant_rescale_factor']

        scope['input_symbol'] = inputs
        scope['output'] = last_x
        scope['model'] = keras.models.Model(inputs, scope['output'])
