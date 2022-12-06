import collections
import os
import json
import flowws
import numpy as np
import pyriodic
from flowws import Argument as Arg


@flowws.add_stage_arguments
class GEOMLoader(flowws.Stage):
    """Load frames from geom database"""

    ARGS = [
        Arg(
            'noise',
            '-n',
            [float],
            [1e-2, 5e-2, 0.1],
            help='Magnitudes of random noise to add to coordinates',
        ),
        Arg(
            'datadir',
            '-d',
            str,
            help='directory where json files exist',
        ),
        Arg(
            'validation_split',
            '-v',
            float,
            .3
        ),
    ]

    Frame = collections.namedtuple(
        'Frame', ['positions', 'box', 'types', 'context'])

    def run(self, scope, storage):
        """run.

        Parameters
        ----------
        scope :
            scope
        storage :
            storage
        """
        train_frames = scope.setdefault('loaded_frames', [])
        val_frames = scope.setdefault('validation_data', [])
        max_types = 0

        # for name in self.arguments['structures']:
        smiles2frame = {}
        for rootdir, subdirs, filenames in os.walk(self.arguments['datadir']):
            for filename in filenames:
                with open(os.path.join(rootdir, filename)) as f:
                    json_obj = json.load(f)
                for item in json_obj:
                    smiles = item['smile']
                    conformer_list = item['conformer_list']
                    smiles2frame[smiles] = conformer_list
        smiles_list = list(smiles2frame.keys())[:1000]
        np.random.seed(42)
        np.random.shuffle(smiles_list)
        val_smiles = smiles_list[:int(len(smiles_list)*self.arguments['validation_split'])]
        train_smiles = smiles_list[int(len(smiles_list)*self.arguments['validation_split']):]

        def add_frames(smiles_list, data_list):
            max_types = 0
            for smile in smiles_list:
                conformers = smiles2frame[smile]
                for conformer in conformers:
                    data_list.append(self.Frame(np.array(conformer['positions']),
                                                np.array(conformer['box']),
                                                np.array(conformer['types']).astype(int),
                                                conformer['context']))
                    max_types = max(max_types, int(np.max(conformer['types'])) + 1)
                    break
            return max_types
        max_types = add_frames(train_smiles, train_frames)
        _ = add_frames(val_smiles, val_frames)
        scope['max_types'] = max_types
