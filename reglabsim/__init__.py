"""F1Lab-AI: Advanced F1 Regulation Stress-Testing Laboratory.

A simulation platform for testing how F1 regulatory changes affect
overtaking, battery dependency, DRS trains, dominance, robustness,
and race strategies.

Example:
    >>> from reglabsim import create_facade
    >>> facade = create_facade()
    >>> result = facade.run_battle_experiment(
    ...     "configs/experiments/baku_closing_speed.yaml"
    ... )
    >>> metrics = facade.compute_metrics(result)
    >>> print(metrics)
    {'dangerous_closing_speed_index': 0.023, 'train_formation_index': 0.18, ...}
"""

__version__ = "0.1.0"

from reglabsim.facade import create_facade

__all__ = [
    "create_facade",
    "__version__",
]