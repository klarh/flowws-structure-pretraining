import functools
import json

import flowws
from flowws import Argument as Arg
import keras_gtar


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
                stages.append(stage)
                print(stage['type'])
            workflow['stages'] = stages
            workflow['scope'] = self.scope
            child_workflow = flowws.Workflow.from_JSON(workflow)
            child_workflow.storage = self.storage
            child_scope = child_workflow.run()
            child_scope.pop('workflow')
            child_scope['model'].set_weights(weights)
        return child_scope
