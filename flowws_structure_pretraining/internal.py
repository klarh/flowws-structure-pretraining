import collections


class Remap:
    def __init__(self):
        self.m = collections.defaultdict(lambda: len(self.m))

    def __call__(self, x):
        return self.m[x]

    @property
    def inverse(self):
        s = sorted([v, k] for (k, v) in self.m.items())
        return [v[1] for v in s]

    def __len__(self):
        return len(self.m)
