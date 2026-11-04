"""Finite or explicitly unlimited update budgets; independent of Torch/RNG."""
from itertools import count


def validate_budget(steps):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError('steps must be nonnegative; 0 means no update limit')


def budget_reached(update, steps):
    validate_budget(steps)
    return steps > 0 and update >= steps


def training_updates(completed, steps):
    validate_budget(steps)
    if completed < 0:
        raise ValueError('completed updates must be nonnegative')
    return count(completed + 1) if steps == 0 else range(completed + 1, steps + 1)


def budget_metadata(steps, source_step=5000):
    validate_budget(steps)
    return dict(additional_updates=steps if steps else None,
                total_compression_updates_target=source_step + steps if steps else None,
                unlimited_training=steps == 0)
