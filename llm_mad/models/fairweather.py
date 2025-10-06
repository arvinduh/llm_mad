# llm_mad/models/fairweather.py

"""Implementation of the 'Fairweather Friend' baseline algorithm."""

import random
from collections.abc import Sequence

from src.llm import classifier

from .base import BanditAlgorithm


class FairweatherFriend(BanditAlgorithm):
  """A simple bandit algorithm that only considers the last review's sentiment.

  This algorithm uses an LLM to classify the previous review as "Good" or "Bad".
  - If "Good", it goes back to that same restaurant.
  - If "Bad", it chooses a different restaurant at random.
  """

  def __init__(
    self, restaurants: Sequence[str], classifier: classifier.ReviewClassifier
  ):
    """Initializes the FairweatherFriend algorithm."""
    super().__init__(restaurants)
    if not classifier:
      raise ValueError("A ReviewClassifier is required.")
    self._classifier = classifier
    self._last_choice: str | None = None
    self._last_review_text: str | None = None

  def select_restaurant(self) -> str:
    """Selects a restaurant based on the last outcome."""
    if self._last_choice and self._last_review_text:
      classification = self._classifier.classify(self._last_review_text)
      if classification == "Good":
        return self._last_choice

    # If it's the first turn or the last review was bad, choose randomly.
    options = [r for r in self.restaurants if r != self._last_choice]
    if not options:
      return self.restaurants[0]

    return random.choice(options)

  def update(self, restaurant: str, review_text: str) -> None:
    """Updates the algorithm with the most recent choice and review text."""
    self._last_choice = restaurant
    self._last_review_text = review_text
