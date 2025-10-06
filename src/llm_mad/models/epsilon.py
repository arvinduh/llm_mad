# llm_mad/models/epsilon_greedy.py

"""Implementation of the Epsilon-Greedy bandit algorithm using classifications."""

import random
from collections.abc import Sequence

from ..llm import classifier
from .base import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
  """An Epsilon-Greedy algorithm that learns from 'Good'/'Bad' classifications.

  This algorithm learns from the history of review classifications for each restaurant.
  - With probability epsilon, it explores by choosing a random restaurant.
  - With probability 1 - epsilon, it exploits by choosing the restaurant with
    the current highest proportion of "Good" reviews.
  """

  def __init__(
    self,
    restaurants: Sequence[str],
    classifier: classifier.ReviewClassifier,
    epsilon: float = 0.1,
  ):
    """Initializes the EpsilonGreedy algorithm.

    Args:
        restaurants: A sequence of restaurant names.
        classifier: The ReviewClassifier used to determine if a review is
          "Good" or "Bad".
        epsilon: The probability of choosing a random arm (exploration).
          Must be between 0.0 and 1.0.
    """
    super().__init__(restaurants)
    if not 0.0 <= epsilon <= 1.0:
      raise ValueError("Epsilon must be between 0.0 and 1.0.")
    if not classifier:
      raise ValueError("A ReviewClassifier is required.")

    self.epsilon = epsilon
    self._classifier = classifier
    # Store counts of 'good' and 'bad' reviews for each restaurant.
    self.counts: dict[str, dict[str, int]] = {
      r: {"good": 0, "bad": 0} for r in self.restaurants
    }
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

    # Exploitation: choose the best-known restaurant based on the proportion
    # of "Good" reviews.
    best_proportion = -1.0
    best_restaurants = []

    for restaurant in self.restaurants:
      counts = self.counts[restaurant]
      total = counts["good"] + counts["bad"]
      if total == 0:
        # Give restaurants with no reviews a very high score to encourage
        # exploration, but lower than a perfect score.
        proportion = 1.0
      else:
        proportion = counts["good"] / total

      if proportion > best_proportion:
        best_proportion = proportion
        best_restaurants = [restaurant]
      elif proportion == best_proportion:
        best_restaurants.append(restaurant)

    # random.choice handles ties by picking one randomly.
    return random.choice(best_restaurants)

  def update(self, restaurant: str, review_text: str) -> None:
    """Updates the counts for the chosen restaurant using the review text."""
    classification = self._classifier.classify(review_text)
    if classification == "Good":
      self.counts[restaurant]["good"] += 1
    else:
      self.counts
