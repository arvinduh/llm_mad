"""Functions for plotting the results of bandit algorithm simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_cumulative_reward(results: pd.DataFrame, title: str):
  """Plots the cumulative reward over time for each algorithm."""
  plt.figure(figsize=(12, 7))
  for algorithm_name, data in results.groupby("algorithm"):
    data["cumulative_score"] = data["score"].cumsum()
    sns.lineplot(
      x="step", y="cumulative_score", data=data, label=algorithm_name
    )
  plt.title(title, fontsize=16)
  plt.xlabel("Time Step")
  plt.ylabel("Cumulative Reward (Score)")
  plt.legend()
  plt.show()


def plot_restaurant_choices(results: pd.DataFrame, title: str):
  """Plots a bar chart of how many times each restaurant was chosen."""
  plt.figure(figsize=(12, 7))
  sns.countplot(x="choice", hue="algorithm", data=results)
  plt.title(title, fontsize=16)
  plt.xlabel("Restaurant")
  plt.ylabel("Number of Times Chosen")
  plt.xticks(rotation=45, ha="right")
  plt.legend(title="Algorithm")
  plt.tight_layout()
  plt.show()


def plot_average_reward_over_time(
  results: pd.DataFrame, title: str, window_size: int = 25
):
  """Plots the rolling average reward over time for each algorithm."""
  plt.figure(figsize=(12, 7))
  for algorithm_name, data in results.groupby("algorithm"):
    data["rolling_avg"] = data["score"].rolling(window=window_size).mean()
    sns.lineplot(
      x="step",
      y="rolling_avg",
      data=data,
      label=f"{algorithm_name} (ws={window_size})",
    )
  plt.title(title, fontsize=16)
  plt.xlabel("Time Step")
  plt.ylabel("Rolling Average Reward")
  plt.legend()
  plt.show()


def plot_ratings_over_time(results: pd.DataFrame, algorithm_name: str):
  """
  Plots the daily ratings and running average over time for a specific algorithm.

  Args:
    results: DataFrame containing simulation results
    algorithm_name: Name of the algorithm to plot
  """
  # Filter data for the specific algorithm
  algorithm_data = results[results["algorithm"] == algorithm_name].copy()
  if algorithm_data.empty:
    print(f"No data found for algorithm: {algorithm_name}")
    return

  # Sort by step to ensure proper ordering
  algorithm_data = algorithm_data.sort_values("step")
  all_ratings = algorithm_data["score"].tolist()

  # Create a list of days for the x-axis
  days_list = list(range(1, len(all_ratings) + 1))

  # Calculate the running average of ratings
  running_average = pd.Series(all_ratings).expanding().mean()

  # Create the plot
  plt.figure(figsize=(12, 6))

  # Plot daily ratings as points
  plt.scatter(days_list, all_ratings, label="Daily Rating", alpha=0.6, s=10)

  # Plot running average as a line
  plt.plot(days_list, running_average, label="Running Average", color="red")

  plt.xlabel("Day")
  plt.ylabel("Rating")
  plt.title(f"Daily Ratings and Running Average Over Time - {algorithm_name}")
  plt.legend()  # Add a legend to distinguish the lines
  plt.grid(True)
  plt.show()


def plot_algorithm_choices_timeline(results: pd.DataFrame, title: str):
  """
  Shows the specific choices that each model made at each time period.

  Args:
    results: DataFrame containing simulation results
    title: Title for the plot
  """
  plt.figure(figsize=(15, 8))

  # Get unique algorithms and restaurants
  algorithms = results["algorithm"].unique()
  restaurants = results["choice"].unique()

  # Create a color map for restaurants
  colors = plt.cm.Set3(range(len(restaurants)))
  restaurant_colors = dict(zip(restaurants, colors))

  # Plot timeline for each algorithm
  for i, algorithm in enumerate(algorithms):
    alg_data = results[results["algorithm"] == algorithm].sort_values("step")

    # Plot points colored by restaurant choice
    for restaurant in restaurants:
      mask = alg_data["choice"] == restaurant
      if mask.any():
        plt.scatter(
          alg_data[mask]["step"],
          [i] * sum(mask),
          c=[restaurant_colors[restaurant]],
          label=restaurant
          if i == 0
          else "",  # Only show legend for first algorithm
          alpha=0.7,
          s=30,
        )

  plt.yticks(range(len(algorithms)), algorithms)
  plt.xlabel("Time Step")
  plt.ylabel("Algorithm")
  plt.title(title, fontsize=16)
  plt.legend(title="Restaurant", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()
