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

    def draw_matplotlib(self, fig):
        ax = fig.add_subplot()
        ax.plot(self.values)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)


@flowws.add_stage_arguments
@flowws.register_module
class LoadModel(flowws.Stage):
    """Load saved model and metadata"""

    ARGS = [
        Arg('filename', '-f', str, help='Names of file to load'),
    ]

    def run(self, scope, storage):
        assert 'filename' in self.arguments
        self.scope = scope
        self.storage = storage
        scope.update(self.child_scope)

        visuals = scope.setdefault('visuals', [])
        for v in self.loaded_visuals:
            if v not in visuals:
                visuals.append(v)

    @functools.cached_property
    def child_scope(self):
        with keras_gtar.Trajectory(self.arguments['filename'], 'r') as traj:
            weights = traj.get_weights()
            workflow = json.loads(traj.handle.readStr('workflow.json'))
            stages = []
            for stage in workflow['stages']:
                if stage['type'] == 'FileLoader':
                    continue
                elif stage['type'] in ('Train', 'Save'):
                    continue
                elif stage['type'] == 'FrameClassificationTask':
                    self.scope['num_classes'] = len(weights[-1])
                stages.append(stage)
                print(stage['type'])
            workflow['stages'] = stages
            workflow['scope'] = self.scope
            child_workflow = flowws.Workflow.from_JSON(workflow)
            child_workflow.storage = self.storage
            child_scope = child_workflow.run()
            child_scope.pop('workflow')
            child_scope['model'].set_weights(weights)

            self.loaded_visuals = self.get_visuals(traj.handle)

        return child_scope

    def get_visuals(self, handle):
        result = []

        records = handle.getRecordTypes()
        continuous_records = [
            r for r in records if r.getBehavior() == gtar.Behavior.Continuous
        ]
        for rec in continuous_records:
            frames = handle.queryFrames(rec)
            value = np.concatenate([handle.getRecord(rec, f) for f in frames])
            result.append(ArrayVisual(value, rec.getName()))
        return result
