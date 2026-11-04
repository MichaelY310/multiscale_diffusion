"""Store inference snapshots only when explicitly requested."""
from .tree import map_tree


class Trace(list):
    def __init__(self, enabled, initial=None):
        super().__init__()
        self.enabled=enabled
        if initial is not None:self.append(initial)

    def append(self,value):
        if self.enabled:
            super().append(map_tree(lambda x:x.detach().clone(),value))
