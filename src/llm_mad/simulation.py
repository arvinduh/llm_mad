# llm_mad/simulation.py

"""Utilities for running and managing bandit algorithm simulations."""

from collections.abc import Sequence

import pandas as pd
from absl import logging
from tqdm.notebook import tqdm

from . import models, reviews
from .llm import ReviewQuantifier


def run_simulation(
  algorithm: models.BanditAlgorithm,
  review_selector: reviews.ReviewSelector,
  quantifier: ReviewQuantifier,
  num_steps: int,
) -> pd.DataFrame:
  """Runs a single simulation for a given bandit algorithm."""
  history = []
  review_selector.reset_all()

  pbar = tqdm(
    range(num_steps),
    desc=f"Simulating {algorithm.__class__.__name__}",
    leave=False,
  )

  for step in pbar:
    chosen_restaurant = algorithm.select_restaurant()

    try:
      review_data = review_selector.get_random_review(chosen_restaurant)
      review_text = str(review_data["Review"])
    except IndexError:
      review_selector.reset(chosen_restaurant)
      review_data = review_selector.get_random_review(chosen_restaurant)
      review_text = str(review_data["Review"])

    # Models that learn from classification use the raw text for their update.
    if isinstance(algorithm, (models.FairweatherFriend, models.EpsilonGreedy)):
      algorithm.update(restaurant=chosen_restaurant, review_text=review_text)
    else:
      # Other models (like RandomChoice) use a quantified score.
      try:
        score = quantifier.quantify(review_text)
      except (RuntimeError, ValueError) as e:
        logging.warning(
          "Step %d: Quantify failed, using original rating. Error: %s", step, e
        )
        score = float(review_data["Rating"])

      algorithm.update(restaurant=chosen_restaurant, score=score)

    # For consistent evaluation, we always record a numeric score for plotting.
    # We use the original star rating here as it's fast and doesn't require
    # another API call if the algorithm didn't already need a quantified score.
    eval_score = float(review_data["Rating"])
    history.append(
      {"step": step, "choice": chosen_restaurant, "score": eval_score}
    )

  return pd.DataFrame(history)


def run_experiment(
  algorithms: Sequence[models.BanditAlgorithm],
  review_selector: reviews.ReviewSelector,
  quantifier: ReviewQuantifier,
  num_steps: int,
) -> pd.DataFrame:
  """Runs a full experiment comparing multiple algorithms."""
  all_results = []
  for algorithm in tqdm(algorithms, desc="Running Full Experiment"):
    results_df = run_simulation(
      algorithm, review_selector, quantifier, num_steps
    )
    results_df["algorithm"] = algorithm.__class__.__name__
    all_results.append(results_df)

  return pd.concat(all_results, ignore_index=True)


def run_synchronized_experiment(
  algorithms: Sequence[models.BanditAlgorithm],
  review_selector: reviews.ReviewSelector,
  quantifier: ReviewQuantifier,
  num_steps: int,
) -> pd.DataFrame:
  """Runs a synchronized experiment where all algorithms use the same timepoints.

  When multiple algorithms choose the same restaurant at the same timestep,
  they get identical results. This ensures fair comparison by eliminating
  the randomness in review selection across algorithms.

  Args:
      algorithms: Sequence of bandit algorithms to compare.
      review_selector: The review selector (will be wrapped in SynchronizedReviewSelector).
      quantifier: The review quantifier for score-based algorithms.
      num_steps: Number of simulation steps to run.

  Returns:
      DataFrame containing results for all algorithms with synchronized timepoints.
  """
  # Create synchronized review selector
  sync_selector = reviews.SynchronizedReviewSelector(
    review_selector._data,
    review_selector._restaurant_col,
    list(review_selector.restaurants),
  )

  all_results = []

  for algorithm in tqdm(algorithms, desc="Running Synchronized Experiment"):
    history = []
    sync_selector.reset_all()

    # Reset algorithm state (create fresh copy)
    algorithm_class = algorithm.__class__
    if hasattr(algorithm, "__dict__"):
      # Create new instance with same parameters
      if hasattr(algorithm, "epsilon") and hasattr(algorithm, "_quantifier"):
        # EpsilonGreedy
        fresh_algorithm = algorithm_class(
          algorithm.restaurants, algorithm._quantifier, algorithm.epsilon
        )
      elif hasattr(algorithm, "_classifier"):
        # FairweatherFriend
        fresh_algorithm = algorithm_class(
          algorithm.restaurants, algorithm._classifier
        )
      else:
        # RandomChoice or other simple algorithms
        fresh_algorithm = algorithm_class(algorithm.restaurants)
    else:
      fresh_algorithm = algorithm

    for step in range(num_steps):
      chosen_restaurant = fresh_algorithm.select_restaurant()

      # Get synchronized review for this timestep and restaurant
      review_data = sync_selector.get_synchronized_review(
        chosen_restaurant, step
      )
      review_text = str(review_data["Review"])

      # Update algorithm based on its type
      if isinstance(fresh_algorithm, models.FairweatherFriend):
        fresh_algorithm.update(
          restaurant=chosen_restaurant, review_text=review_text
        )
      else:
        # Other models (like RandomChoice and EpsilonGreedy) use a quantified score
        try:
          score = quantifier.quantify(review_text)
        except (RuntimeError, ValueError) as e:
          logging.warning(
            "Step %d: Quantify failed, using original rating. Error: %s",
            step,
            e,
          )
          score = float(review_data["Rating"])

        fresh_algorithm.update(restaurant=chosen_restaurant, score=score)

      # Record the result
      eval_score = float(review_data["Rating"])
      history.append(
        {"step": step, "choice": chosen_restaurant, "score": eval_score}
      )

    # Convert to DataFrame and add algorithm name
    results_df = pd.DataFrame(history)
    results_df["algorithm"] = algorithm.__class__.__name__
    all_results.append(results_df)

  return pd.concat(all_results, ignore_index=True)


def create_synchronized_experiment(
  algorithms: Sequence[models.BanditAlgorithm],
  data: pd.DataFrame,
  quantifier: ReviewQuantifier,
  num_steps: int,
  restaurant_col: str = "Restaurant",
  restaurants: Sequence[str] | None = None,
) -> pd.DataFrame:
  """Convenience function to create and run a synchronized experiment from data.

  Args:
      algorithms: Sequence of bandit algorithms to compare.
      data: DataFrame containing review data.
      quantifier: The review quantifier for score-based algorithms.
      num_steps: Number of simulation steps to run.
      restaurant_col: Name of the restaurant column in data.
      restaurants: Optional subset of restaurants to include.

  Returns:
      DataFrame containing synchronized experimental results.
  """
  sync_selector = reviews.SynchronizedReviewSelector(
    data, restaurant_col, restaurants
  )

  return run_synchronized_experiment(
    algorithms, sync_selector, quantifier, num_steps
  )
