"""Defines the abstract base class for a multi-armed bandit algorithm."""

import abc
from collections.abc import Sequence


class BanditAlgorithm(abc.ABC):
  """Abstract base class for a multi-armed bandit algorithm."""

  def __init__(self, restaurants: Sequence[str]):
    """Initializes the algorithm.

    Args:
        restaurants: A sequence of restaurant names (the "arms").
    """
    if not restaurants:
      raise ValueError("Restaurant list cannot be empty.")
    self.restaurants = list(restaurants)

  @abc.abstractmethod
  def select_restaurant(self) -> str:
    """Selects a restaurant to visit.

    Returns:
        The name of the chosen restaurant.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def update(self, restaurant: str, score: int | float) -> None:
    """Updates the algorithm's knowledge with a new review score.

    Args:
        restaurant: The name of the restaurant that was reviewed.
        score: The numerical score of the review.
    """
    raise NotImplementedError
