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
