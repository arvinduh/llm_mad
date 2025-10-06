"""Module for selecting random restaurant reviews without replacement."""

import random
from collections.abc import Sequence

import pandas as pd


class ReviewSelector:
  """Selects random reviews for restaurants without replacement.

  This class pre-processes a DataFrame of reviews to allow for efficient,
  non-repeating random selection of reviews for specified restaurants. Once all
  reviews for a restaurant have been selected, subsequent requests will raise
  an IndexError until the selections for that restaurant are reset.

  Attributes:
      restaurants: A frozenset of the restaurant names available for selection.
  """

  def __init__(
    self,
    data: pd.DataFrame,
    restaurant_col: str = "Restaurant",
    restaurants: Sequence[str] | None = None,
  ) -> None:
    """Initializes the ReviewSelector.

    Args:
        data: A DataFrame containing the reviews.
        restaurant_col: The name of the column containing restaurant names.
        restaurants: An optional sequence of restaurant names to include. If
          None, all unique restaurants from the data are included.
    """
    self._restaurant_col = restaurant_col

    if restaurants is None:
      self._data: pd.DataFrame = data.copy()
      self._restaurants: frozenset[str] = frozenset(
        data[self._restaurant_col].unique()
      )
    else:
      self._restaurants = frozenset(restaurants)
      self._data = data[
        data[self._restaurant_col].isin(self._restaurants)
      ].copy()

    self._review_indices: dict[str, list[int]] = self._build_indices()
    self._shuffled_indices: dict[str, list[int]] = {}

  def _build_indices(self) -> dict[str, list[int]]:
    """Builds a map from restaurant name to a list of its review indices."""
    grouped = self._data.groupby(self._restaurant_col)
    return {str(name): group.index.tolist() for name, group in grouped}

  @property
  def restaurants(self) -> frozenset[str]:
    """Returns the immutable set of available restaurant names."""
    return self._restaurants

  def get_random_review(self, restaurant: str) -> pd.Series:
    """Returns a random, not-yet-seen review for the given restaurant.

    Each call for a given restaurant is guaranteed to return a unique review
    until all reviews for that restaurant have been returned.

    Args:
        restaurant: The name of the restaurant.

    Returns:
        A pandas Series representing a single review row.

    Raises:
        ValueError: If the restaurant name is not found in the selector.
        IndexError: If all reviews for the restaurant have already been returned.
    """
    if restaurant not in self._restaurants:
      raise ValueError(f"Restaurant '{restaurant}' not found.")

    if restaurant not in self._review_indices:
      raise IndexError(f"No reviews available for restaurant '{restaurant}'.")

    if restaurant not in self._shuffled_indices:
      # First request: Create and shuffle a copy of its review indices.
      indices = self._review_indices[restaurant][:]
      random.shuffle(indices)
      self._shuffled_indices[restaurant] = indices

    available_indices = self._shuffled_indices[restaurant]
    if not available_indices:
      raise IndexError(
        f"All reviews for restaurant '{restaurant}' have been selected."
      )

    review_index = available_indices.pop()
    return self._data.loc[review_index]

  def reset(self, restaurant: str) -> None:
    """Resets the pool of available reviews for a specific restaurant.

    After calling this, `get_random_review` will once again be able to return
    any review for the specified restaurant.

    Args:
        restaurant: The name of the restaurant to reset.

    Raises:
        ValueError: If the restaurant name is not found.
    """
    if restaurant not in self._restaurants:
      raise ValueError(f"Restaurant '{restaurant}' not found.")
    self._shuffled_indices.pop(restaurant, None)

  def reset_all(self) -> None:
    """Resets the pool of available reviews for all restaurants."""
    self._shuffled_indices.clear()
