"""Implementation of the Epsilon-Greedy bandit algorithm."""

import random
from collections.abc import Sequence

from .base import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
  """An Epsilon-Greedy multi-armed bandit algorithm.

  This algorithm learns from all past reviews for each restaurant.
  - With probability epsilon, it explores by choosing a random restaurant.
  - With probability 1 - epsilon, it exploits by choosing the restaurant with
    the current highest average score.
  """

  def __init__(self, restaurants: Sequence[str], epsilon: float = 0.1):
    """Initializes the EpsilonGreedy algorithm.

    Args:
        restaurants: A sequence of restaurant names.
        epsilon: The probability of choosing a random arm (exploration).
          Must be between 0.0 and 1.0.
    """
    super().__init__(restaurants)
    if not 0.0 <= epsilon <= 1.0:
      raise ValueError("Epsilon must be between 0.0 and 1.0.")
    self.epsilon = epsilon
    self.scores: dict[str, list[int | float]] = {
      r: [] for r in self.restaurants
    }
    self.averages: dict[str, float] = {r: 0.0 for r in self.restaurants}
    # A counter to ensure each restaurant is selected at least once initially.
    self._initial_rounds_left = list(self.restaurants)

  def select_restaurant(self) -> str:
    """Selects a restaurant using the epsilon-greedy strategy.

    It first ensures each restaurant is tried once before applying the
    epsilon-greedy logic.

    Returns:
        The name of the chosen restaurant.
    """
    # Ensure each restaurant is selected at least once at the start.
    if self._initial_rounds_left:
      return self._initial_rounds_left.pop(0)

    if random.random() < self.epsilon:
      # Exploration: choose a random restaurant.
      return random.choice(self.restaurants)
    else:
      # Exploitation: choose the best-known restaurant.
      # random.choice handles ties by picking one randomly.
      best_score = -1.0
      best_restaurants = []
      for restaurant, avg_score in self.averages.items():
        if avg_score > best_score:
          best_score = avg_score
          best_restaurants = [restaurant]
        elif avg_score == best_score:
          best_restaurants.append(restaurant)
      return random.choice(best_restaurants)

  def update(self, restaurant: str, score: int | float) -> None:
    """Updates the scores and average for the chosen restaurant.

    Args:
        restaurant: The name of the restaurant that was reviewed.
        score: The numerical score of the review.
    """
    self.scores[restaurant].append(score)
    # Re-calculate the average
    self.averages[restaurant] = sum(self.scores[restaurant]) / len(
      self.scores[restaurant]
    )
