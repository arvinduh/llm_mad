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
