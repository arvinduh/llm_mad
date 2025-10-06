# llm_mad/models/epsilon_greedy.py

"""Implementation of the Epsilon-Greedy bandit algorithm using classifications."""

import random
from collections.abc import Sequence

from ..llm import quantifier
from .base import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
  """An Epsilon-Greedy algorithm that learns from numerical review scores.

  This algorithm learns from the history of review scores for each restaurant.
  - With probability epsilon, it explores by choosing a random restaurant.
  - With probability 1 - epsilon, it exploits by choosing the restaurant with
    the current highest average score.
  """

  def __init__(
    self,
    restaurants: Sequence[str],
    quantifier: quantifier.ReviewQuantifier,
    epsilon: float = 0.1,
  ):
    """Initializes the EpsilonGreedy algorithm.

    Args:
        restaurants: A sequence of restaurant names.
        quantifier: The ReviewQuantifier used to convert reviews to numerical scores.
        epsilon: The probability of choosing a random arm (exploration).
          Must be between 0.0 and 1.0.
    """
    super().__init__(restaurants)
    if not 0.0 <= epsilon <= 1.0:
      raise ValueError("Epsilon must be between 0.0 and 1.0.")
    if not quantifier:
      raise ValueError("A ReviewQuantifier is required.")

    self.epsilon = epsilon
    self._quantifier = quantifier
    # Store sum of scores and count of reviews for each restaurant.
    self.scores: dict[str, list[float]] = {r: [] for r in self.restaurants}
    # A counter to ensure each restaurant is selected at least once initially.
    self._initial_rounds_left = list(self.restaurants)

  def select_restaurant(self) -> str:
    """Selects a restaurant using the epsilon-greedy strategy.

    It first ensures each restaurant is tried once before applying the
    epsilon-greedy logic.

    Returns:
        The name of the chosen restaurant.
    """
    if self._initial_rounds_left:
      return self._initial_rounds_left.pop(0)

    if random.random() < self.epsilon:
      # Exploration: choose a random restaurant.
      return random.choice(self.restaurants)

    # Exploitation: choose the best-known restaurant based on average score.
    best_average = -float("inf")
    best_restaurants = []

    for restaurant in self.restaurants:
      scores = self.scores[restaurant]
      if len(scores) == 0:
        # Give restaurants with no reviews a high score to encourage
        # exploration, but not infinite.
        average = 5.0  # Assume 5-star scale
      else:
        average = sum(scores) / len(scores)

      if average > best_average:
        best_average = average
        best_restaurants = [restaurant]
      elif average == best_average:
        best_restaurants.append(restaurant)

    # random.choice handles ties by picking one randomly.
    return random.choice(best_restaurants)

  def update(self, restaurant: str, score: int | float) -> None:
    """Updates the algorithm's knowledge with a new review score.

    Args:
        restaurant: The name of the restaurant that was reviewed.
        score: The numerical score of the review.
    """
    self.scores[restaurant].append(float(score))
