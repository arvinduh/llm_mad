"""Utilities for running and managing bandit algorithm simulations."""

from collections.abc import Sequence

import pandas as pd
from tqdm.notebook import tqdm

from src import quantify, reviews
from src.models import base


def run_simulation(
  algorithm: base.BanditAlgorithm,
  review_selector: reviews.ReviewSelector,
  quantifier: quantify.ReviewQuantifier,
  num_steps: int,
) -> pd.DataFrame:
  """Runs a single simulation for a given bandit algorithm."""
  history = []
  # Ensure the selector is reset for a fair run
  review_selector.reset_all()

  # Progress bar setup
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
      score = quantifier.quantify(review_text)
    except IndexError:
      review_selector.reset(chosen_restaurant)
      review_data = review_selector.get_random_review(chosen_restaurant)
      review_text = str(review_data["Review"])
      score = quantifier.quantify(review_text)
    except (RuntimeError, ValueError) as e:
      print(
        f"Step {step}: Could not quantify, assigning neutral score 50. Error: {e}"
      )
      score = 50

    algorithm.update(chosen_restaurant, score)
    history.append({"step": step, "choice": chosen_restaurant, "score": score})

  return pd.DataFrame(history)


def run_experiment(
  algorithms: Sequence[base.BanditAlgorithm],
  review_selector: reviews.ReviewSelector,
  quantifier: quantify.ReviewQuantifier,
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
