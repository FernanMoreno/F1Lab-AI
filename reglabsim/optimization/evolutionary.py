"""Evolutionary optimization."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class EvolutionaryOptimizer:
    """Evolutionary algorithm for optimization.

    Genetic algorithm with mutation and crossover.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize optimizer."""
        self._seed = seed

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, tuple],
        n_generations: int = 100,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ) -> Dict[str, Any]:
        """Run evolutionary optimization.

        Args:
            objective_fn: Objective to minimize.
            search_space: Dict of parameter bounds.
            n_generations: Number of generations.
            population_size: Population size.
            mutation_rate: Mutation probability.
            crossover_rate: Crossover probability.

        Returns:
            Optimization result.
        """
        import numpy as np

        rng = np.random.default_rng(self._seed)

        # Initialize population
        population = self._init_population(search_space, population_size, rng)

        best_value = float("inf")
        best_individual = None

        for gen in range(n_generations):
            # Evaluate fitness
            fitness = [objective_fn(ind) for ind in population]

            # Track best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_value:
                best_value = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            # Selection - keep best half
            sorted_indices = np.argsort(fitness)
            survivors = [population[i] for i in sorted_indices[:population_size // 2]]

            # Create next generation
            next_gen = survivors.copy()
            while len(next_gen) < population_size:
                # Crossover
                if rng.random() < crossover_rate and len(survivors) >= 2:
                    parent1, parent2 = rng.choice(survivors, 2, replace=False)
                    child = self._crossover(parent1, parent2, rng)
                else:
                    child = rng.choice(survivors).copy()

                # Mutation
                if rng.random() < mutation_rate:
                    child = self._mutate(child, search_space, rng)

                next_gen.append(child)

            population = next_gen

        return {
            "best_params": best_individual,
            "best_value": best_value,
            "generations": n_generations,
        }

    def _init_population(
        self,
        search_space: Dict[str, tuple],
        size: int,
        rng,
    ) -> List[Dict[str, float]]:
        """Initialize random population."""
        population = []
        for _ in range(size):
            individual = {
                name: rng.uniform(bounds[0], bounds[1])
                for name, bounds in search_space.items()
            }
            population.append(individual)
        return population

    def _crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float],
        rng,
    ) -> Dict[str, float]:
        """Crossover two parents."""
        child = {}
        for key in parent1:
            child[key] = (parent1[key] + parent2[key]) / 2
        return child

    def _mutate(
        self,
        individual: Dict[str, float],
        search_space: Dict[str, tuple],
        rng,
    ) -> Dict[str, float]:
        """Mutate individual."""
        mutated = individual.copy()
        for key in mutated:
            low, high = search_space[key]
            mutated[key] += rng.normal(0, (high - low) * 0.05)
            mutated[key] = max(low, min(high, mutated[key]))
        return mutated