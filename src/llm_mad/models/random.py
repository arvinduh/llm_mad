# llm_mad/models/random_choice.py

"""Implementation of the 'Random Choice' baseline algorithm."""

import random
from collections.abc import Sequence

from .base import BanditAlgorithm


class RandomChoice(BanditAlgorithm):
  """A simple baseline algorithm that chooses a restaurant randomly."""

  def __init__(self, restaurants: Sequence[str]):
    """Initializes the RandomChoice algorithm."""
    super().__init__(restaurants)

  def select_restaurant(self) -> str:
    """Selects a restaurant completely at random."""
    return random.choice(self.restaurants)

  def update(self, restaurant: str, score: int | float) -> None:
    """Does nothing, as this algorithm does not learn."""
    pass
