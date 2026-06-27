"""Training algorithm registry."""

from gumbel_az.algorithms.registry import create_algorithm, registered_algorithms

__all__ = ["create_algorithm", "registered_algorithms"]
