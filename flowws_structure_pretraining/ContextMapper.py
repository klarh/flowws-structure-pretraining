import flowws
from flowws import Argument as Arg


@flowws.add_stage_arguments
class ContextMapper(flowws.Stage):
    """Modify frame contexts with code snippets."""

    ARGS = [
        Arg('script', '-s', str, 'f = lmabda context: context', help='Code to be run'),
        Arg(
            'function_name',
            '-f',
            str,
            'f',
            help='Name of mapping function in the script',
        ),
    ]

    def run(self, scope, storage):
        function_scope = {}
        exec(self.arguments['script'], function_scope)

        try:
            map_function = function_scope[self.arguments['function_name']]
        except KeyError:
            raise ValueError(
                'Function {} not present after evaluating the code:\n'.format(
                    self.arguments['function_name'], self.arguments['script']
                )
            )

        frames = []
        for frame in scope['loaded_frames']:
            frames.append(frame._replace(context=map_function(frame.context)))
        scope['loaded_frames'] = frames
