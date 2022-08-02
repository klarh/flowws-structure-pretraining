import flowws


class ClearMetrics(flowws.Stage):
    """Clear out any loaded metrics for calculation"""

    def run(self, scope, storage):
        scope.pop('metrics', [])
