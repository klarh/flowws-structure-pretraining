import functools
import json

import flowws
from flowws import Argument as Arg
import gtar
import keras_gtar
import numpy as np


class ArrayVisual:
    def __init__(self, values, ylabel, xlabel='epoch'):
        self.values = values
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.visual_name = 'plot.{}'.format(ylabel)

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()
        ax.plot(self.values)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)


@flowws.add_stage_arguments
class LoadModel(flowws.Stage):
    """Load saved model and metadata"""

    ARGS = [
        Arg('filename', '-f', str, help='Names of file to load'),
        Arg(
            'subsample',
            '-s',
            float,
            help='Override the subsampling operation of task modules loaded here',
        ),
        Arg(
            'disable_shuffle',
            None,
            bool,
            False,
            help='Disable shuffling of data if True',
        ),
        Arg(
            'only_model',
            '-m',
            bool,
            False,
            help='If True, only load the model, not any other data modules',
        ),
        Arg(
            'no_model',
            '-n',
            bool,
            False,
            help='If True, do not perform logic related to tensorflow/creating the model',
        ),
    ]

    def run(self, scope, storage):
        assert 'filename' in self.arguments or 'model_filename' in scope
        self.scope = scope
        self.storage = storage
        scope.update(self.child_scope)

        visuals = scope.setdefault('visuals', [])
        for v in self.loaded_visuals:
            if v not in visuals:
                visuals.append(v)

    @functools.cached_property
    def child_scope(self):
        filename = self.arguments.get(
            'filename', self.scope.get('model_filename', None)
        )
        with keras_gtar.Trajectory(filename, 'r') as traj:
            if not self.arguments['no_model']:
                weights = traj.get_weights()
            workflow = json.loads(traj.handle.readStr('workflow.json'))
            stages = []
            for stage in workflow['stages']:
                if stage['type'] in ('FileLoader', 'PyriodicLoader'):
                    continue
                elif stage['type'] in ('Train', 'Save'):
                    continue
                elif stage['type'] == 'InitializeTF':
                    if self.arguments['no_model']:
                        continue
                elif any(
                    stage['type'].endswith(bit)
                    for bit in ('Classifier', 'Regressor', 'Autoencoder')
                ):
                    if self.arguments['no_model']:
                        continue

                if stage['type'] == 'FrameClassificationTask':
                    if not self.arguments['no_model']:
                        self.scope['num_classes'] = len(weights[-1])
                        self.scope['freeze_num_classes'] = True
                if stage['type'].endswith('Task') and 'subsample' in self.arguments:
                    stage['arguments']['subsample'] = self.arguments['subsample']

                stages.append(stage)
                print(stage['type'])
            workflow['stages'] = stages
            workflow['scope'] = self.scope
            child_workflow = flowws.Workflow.from_JSON(workflow)
            for stage in child_workflow.stages:
                if (
                    'shuffle' in stage.arg_specifications
                    and self.arguments['disable_shuffle']
                ):
                    stage.arguments['shuffle'] = False
            child_workflow.storage = self.storage
            if self.arguments['only_model']:
                model = traj.load()
                child_scope = dict(model=model)
                max_types = model.inputs[1].get_shape().as_list()[-1] // 2
                child_scope['max_types'] = max_types
            else:
                child_scope = child_workflow.run()
                child_scope.pop('workflow')
                child_scope.pop('metadata')
                if not self.arguments['no_model']:
                    child_scope['model'].set_weights(weights)
            child_scope['preprocess_workflow'] = child_workflow
            self.loaded_visuals = self.get_visuals(traj.handle)

        self.scope['model_filename'] = filename
        return child_scope

    def get_visuals(self, handle):
        result = []

        records = handle.getRecordTypes()
        continuous_records = [
            r for r in records if r.getBehavior() == gtar.Behavior.Continuous
        ]
        for rec in continuous_records:
            frames = handle.queryFrames(rec)
            try:
                value = np.concatenate([handle.getRecord(rec, f) for f in frames])
                result.append(ArrayVisual(value, rec.getName()))
            except ValueError:  # can't concatenate, probably not a numeric-type array
                pass
        return result
