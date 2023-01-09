import collections
import os
import json
import flowws
import numpy as np
import pyriodic
from flowws import Argument as Arg
from tqdm import tqdm
import msgpack
from tqdm import tqdm
import psutil

TARGET_SMILES=set(["C1=CC=C2C=CC=CC2=C1", 
        "C1=CNC(=O)NC1=O",
        "C1=CC=CC=C1",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "C1=CC=C(C(=C1)C(=O)O)O",
        "C(C=O)C=O",
        "CCO",
        "CC1=CC=CC=C1"])
print(TARGET_SMILES)

def process_iter():
    Frame = collections.namedtuple(
        'Frame', ['positions', 'box', 'types', 'context'])

    direc='/scratch/ssd002/datasets/GEOM/'
    drugs_file = os.path.join(direc, 'drugs_crude.msgpack')
    drugs_file = os.path.join(direc, 'qm9_crude.msgpack')
    unpacker = msgpack.Unpacker(open(drugs_file, 'rb')) # iterator for 292 dictionaries, each containing ~1000 molecules

    full_list = {}
    smiles_list = []
    max_types=0
    for ii, group in tqdm(enumerate(iter(unpacker))):
        #full_list = []
        if ii >= 5: break
        smiles = list(group.keys())
        for smile in smiles:
            conformer_dict = group[smile]
            conformers = conformer_dict['conformers']
            conformer_list = []
            for conformer in conformers:
                conformer['xyz'] = np.array(conformer['xyz'])
                types = conformer['xyz'][:, 0]
                positions = conformer['xyz'][:, 1:]

                Lx = np.max(positions[:, 0]) - np.min(positions[:, 0])+1
                Ly = np.max(positions[:, 1]) - np.min(positions[:, 1])+1
                Lz = np.max(positions[:, 2]) - np.min(positions[:, 2])+1
                if Lx ==0 or Ly == 0 or Lz ==0:
                    print(positions)
                    print(0, smile, ii); import sys; sys.exit(0)
                box = [np.ceil(Lx * 16), np.ceil(Ly * 16), np.ceil(Lz * 16), 0, 0, 0]

                context = {"bw": np.float32(conformer['boltzmannweight']),
                           "te": np.float32(conformer['totalenergy']),
                           "relativeenergy": np.float32(conformer['relativeenergy']),
                           "geom_id": np.float32(conformer['geom_id'])}
                reformatted_dict = {"name": smile,
                                   "context": context,
                                   "box": np.array(box).astype(int),
                                   "positions": np.array(positions, dtype="float32"),
                                   "space_group": -1,
                                   "types": np.array(types).astype(int)
                                   }
                conformer = reformatted_dict
                conformer_list.append(Frame(conformer['positions'],
                                                conformer['box'],
                                                conformer['types'],
                                                conformer['context']))
                max_types = max(max_types, int(np.max(conformer['types'])) + 1)
                #conformer_list.append(reformatted_dict)
            full_list[smile] = conformer_list
            smiles_list.append(smile)
            del conformer_list
            del conformers
            del conformer_dict
        del group
    del unpacker
    return full_list, smiles_list,max_types

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
        print("Loading files.....")
        if False:
            smiles2frame = {}
            for rootdir, subdirs, filenames in os.walk(self.arguments['datadir']):
                for filename in tqdm(filenames):
                    with open(os.path.join(rootdir, filename)) as f:
                        json_obj = json.load(f)
                    for item in json_obj:
                        smiles = item['smile']
                        conformer_list = item['conformer_list']
                        smiles2frame[smiles] = conformer_list
        smiles2frame, smiles_list, max_types = process_iter()
        print("Done loading files!")
        np.random.seed(42)
        np.random.shuffle(smiles_list)
        val_smiles = smiles_list[:int(len(smiles_list)*self.arguments['validation_split'])]
        train_smiles = smiles_list[int(len(smiles_list)*self.arguments['validation_split']):]
        print(f"{len(train_smiles)} molecules in train set")
        print(f"{len(val_smiles)} molecules in val set")
        print(f"{len(smiles_list)} total molecules")
        def add_frames(smiles_list, data_list):
            max_types = 0
            print("i am here 127")
            for ii, smile in enumerate(smiles_list):
                conformers = smiles2frame[smile]
                for conformer in conformers:
                    data_list.append(conformer)
                    #data_list.append(self.Frame(conformer['positions'],
                    #                            conformer['box'],
                    #                            conformer['types'],
                    #                            conformer['context']))
                    #max_types = max(max_types, int(np.max(conformer['types'])) + 1)
              #  del smiles2frame[smile]
              #  del conformers
            #return max_types
        #max_types = add_frames(train_smiles, train_frames)
        add_frames(train_smiles, train_frames)
        #k_ = add_frames(val_smiles, val_frames)
        add_frames(val_smiles, val_frames)
        print(f"{len(train_frames)} conformers in train set")
        print(f"{len(val_frames)} conformers in val set")
        print(f"{len(train_frames)  + len(val_frames)} total conformers")
        scope['max_types'] = max_types
