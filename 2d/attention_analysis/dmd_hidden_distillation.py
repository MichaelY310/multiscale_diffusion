"""Training-only, on-policy hidden distillation for the 24-block DMD student.

No adapter or change to inference. Indices here are zero-based; displayed block
numbers are one-based. Teachers must see the student's detached stage INPUT,
not the frozen teacher's own rollout. The loss also detaches every target.
"""
import math
import torch
from torch.nn import functional as F


HIDDEN_LAYERS = {1: (23,), 2: (21, 23)}
TERM_NAMES = ('middle_block24', 'fine_block22_lowpass', 'fine_block24')


def hidden_config(weight, ramp_steps):
    if not math.isfinite(weight) or weight < 0 or ramp_steps < 0:
        raise ValueError('Hidden weight must be finite/nonnegative; ramp must be nonnegative')
    if weight == 0:
        return {'enabled': False}
    return dict(enabled=True, weight=float(weight), ramp_steps=int(ramp_steps),
                version=1, terms=list(TERM_NAMES), teacher='original_frozen_3NFE_DMD',
                teacher_input='student_stage_input_detached', normalization='per_example_teacher_power',
                fine_block22_pool_factor=2, added_inference_parameters=0)


def hidden_weight_at(update, weight, ramp_steps):
    hidden_config(weight, ramp_steps)
    if update < 0:
        raise ValueError('Update must be nonnegative')
    return weight * (min(1., update / ramp_steps) if ramp_steps else 1.)


def pool_hidden(hidden, factor=2):
    """Average adjacent token cells, NOT channels or RGB image pixels."""
    if hidden.ndim != 3 or factor < 1:
        raise ValueError('Expected [batch, square_grid_tokens, channels] and positive factor')
    b, n, c = hidden.shape
    g = math.isqrt(n)
    if g*g != n or g % factor:
        raise ValueError('Hidden token grid must be square and divisible by the pooling factor')
    return F.avg_pool2d(hidden.float().transpose(1,2).reshape(b,c,g,g),factor).flatten(2).transpose(1,2)


def normalized_hidden_mse(student, teacher, eps=1e-6):
    """Average of per-example relative squared errors, with a fixed power floor.

    No normalization of student amplitude: collapsing/scaling the student is
    penalized. FP32 loss math; teacher is always stop-gradient.
    """
    if student.shape != teacher.shape or student.ndim != 3 or eps <= 0:
        raise ValueError('Hidden tensors must have matching [B,N,C] shapes and eps>0')
    ref = teacher.detach().float()
    return ((student.float()-ref).square().mean((1,2)) /
            ref.square().mean((1,2)).clamp_min(eps)).mean()


def hidden_distillation_loss(student_hidden, teacher_hidden):
    pairs = [(student_hidden[1][23], teacher_hidden[1][23]),
             (pool_hidden(student_hidden[2][21]), pool_hidden(teacher_hidden[2][21].detach())),
             (student_hidden[2][23], teacher_hidden[2][23])]
    terms = {name: normalized_hidden_mse(s,t) for name,(s,t) in zip(TERM_NAMES,pairs)}
    total = torch.stack(list(terms.values())).mean()
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite hidden distillation loss')
    return total, terms
