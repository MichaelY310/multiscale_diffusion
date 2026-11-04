"""Explicit fine-NFE schedules. Independent of, and never changes, old samplers.

Indices are zero-based calls in the 32x32 stage, NOT global diffusion timesteps.
An inactive call uses the complete ORIGINAL teacher, not a hybrid backbone.
"""


def schedules():
    return {
        'original_dense': (),
        'all90': tuple(range(90)),
        'early75': tuple(range(75)),
        'early60': tuple(range(60)),
        'late60': tuple(range(30, 90)),
        'periodic60': tuple(i for i in range(90) if i % 3 != 2),
        'sandwich60': tuple(range(15, 75)),
        'periodic45': tuple(range(0, 90, 2)),
        'early30': tuple(range(30)),
    }


def validate_indices(indices):
    values = tuple(indices)
    if len(set(values)) != len(values) or any(type(i) is not int or i not in range(90) for i in values):
        raise ValueError('Fine NFE indices must be unique integers in [0,89]')
    return frozenset(values)


class ScheduledPrefixRollout:
    def __init__(self, teacher, student, indices, manual=None):
        self.indices = validate_indices(indices)
        if self.indices and student is None:
            raise ValueError('Active schedule requires a student')
        if manual is None:
            from sample_csra_adapter_preview import manual_forward
            manual = manual_forward
        self.manual = manual
        self.teacher, self.student = teacher, student
        self.cache, self.labels = None, None
        self.endpoints = {}
        self.fine_calls = self.reused_fine_calls = 0
        self.fine_timesteps, self.active_timesteps = [], []

    def forward(self, x, t, y):
        if self.labels is None:
            self.labels = y.detach().cpu().clone()
        if not bool((t == t[0]).all()):
            raise ValueError('RDD sampler expected a shared timestep within each batch')
        r, step = int(x.shape[-1]), int(t[0])
        if r == 32:
            index = self.fine_calls
            self.fine_calls += 1
            self.fine_timesteps.append(step)
            if index >= 90:
                raise RuntimeError('Unexpected extra fine-stage call')
        if r == 16 and step == 300 and self.indices:
            out, trace = self.manual(self.teacher, x, t, y, multires=True, capture_layers={13})
            self.cache = {13: trace[13]['output']}
        elif r == 32 and index in self.indices:
            if self.cache is None:
                raise RuntimeError('Missing ORIGINAL 16x16,t300,layer13 hidden')
            out = self.student({'x': x, 't': t, 'labels': y, 'coarse': self.cache})
            self.reused_fine_calls += 1
            self.active_timesteps.append(step)
        else:
            out, _ = self.manual(self.teacher, x, t, y, multires=True)
        if (r, step) in ((8, 600), (16, 300)):
            self.endpoints[str(r)] = {'input': x.detach().cpu().clone(), 'output': out.detach().cpu().clone()}
        return out
