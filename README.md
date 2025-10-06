# LLM-Powered Multi-Armed Bandits for Restaurant Reviews

Welcome! This project provides all the code you need to run a simulation comparing two "multi-armed bandit" algorithms. These algorithms help a computer learn which restaurant is the best by trying them out and analyzing LLM-scored reviews.

This guide will walk you through setting up and running the entire experiment in Google Colab, step by step.

[TOC]

---

## Prerequisites

Before you begin, make sure you have these three things ready:

1.  **A Google Account**: To use Google Colab.
2.  **Your GitHub Repository URL**: The URL you use to access your project files on GitHub.
3.  **An OpenRouter API Key**: This is a free key that lets your code use powerful language models (like Gemini) to score the reviews.
    - Go to [OpenRouter.ai](https://openrouter.ai/keys) to sign up and get your key.
    - It's a long string of letters and numbers, like `sk-or-v1-abc...xyz`. Keep it safe!

---

## Step-by-Step Colab Setup Guide

Follow these steps exactly to run your simulation.

### Step 1: Open a New Colab Notebook

First, go to [https://colab.research.google.com/](https://colab.research.google.com/) and create a new notebook.

### Step 2: Clone Your Project Repository

In the very first code cell of your notebook, you need to download your project files from GitHub.

- The second line, `%cd llm_mad`, moves us into the newly downloaded project folder.

```python
# In a code cell, run this:
!git clone https://github.com/arvinduh/llm_mad.git llm_mad
%cd llm_mad
```

### Step 3: Install the Project

Next, we need to install the project's code and all the libraries it depends on (like `pandas` and `matplotlib`). This command handles everything for you.

```python
# In a new code cell, run this:
!pip install -e . -r requirements.txt --quiet
```

### Step 4: Securely Add Your API Key

It is **very important** not to paste your API key directly into a code cell. We'll use Colab's built-in "Secrets" manager, which keeps it safe.

1.  Click the **key icon** (🔑) on the left sidebar of your Colab notebook.
2.  Click **`+ Add a new secret`**.
3.  For the **Name**, enter `OPENROUTER_API_KEY`. (It must be exactly this name).
4.  For the **Value**, paste your API key from the OpenRouter website.
5.  Click the checkbox to allow notebook access to this secret.

### Step 5: Run the Full Experiment!

This is the final step! Copy the entire code block below into a new cell in your notebook and run it.

This single block of code will:

1.  Load your API key from the Secrets manager.
2.  Load the restaurant review data.
3.  Set up the algorithms.
4.  Run the full simulation (this may take a few minutes).
5.  Print the final scores and generate all the plots for your analysis.

```python
# In a final code cell, copy and paste all of this:

# --- Imports ---
import pandas as pd
from google.colab import userdata

# Import our project's modules
from llm_mad import models, quantify, reviews, simulation, visualization

# --- Load API Key ---
try:
    OPENROUTER_API_KEY = userdata.get('OPENROUTER_API_KEY')
    print("OpenRouter API key loaded successfully.")
except Exception:
    print("Secret 'OPENROUTER_API_KEY' not found. Check your spelling in the Secrets manager.")

# --- Configuration ---
DATA_PATH = 'data/reviews.csv'
NUM_SIMULATION_STEPS = 200
EPSILON = 0.1

# --- Initialization ---
print("\nLoading data...")
df_reviews = pd.read_csv(DATA_PATH)
df_reviews.dropna(subset=['Review', 'Restaurant'], inplace=True)
all_restaurants = list(df_reviews['Restaurant'].unique())
print(f"Found {len(all_restaurants)} unique restaurants.")

# Instantiate our tools
review_selector = reviews.ReviewSelector(df_reviews)
quantifier = quantify.ReviewQuantifier(
    api_key=OPENROUTER_API_KEY, model=quantify.Model.GEMINI_FLASH
)

# Create a list of the algorithms we want to compare
algorithms_to_run = [
    models.EpsilonGreedy(restaurants=all_restaurants, epsilon=EPSILON),
    models.FairweatherFriend(restaurants=all_restaurants),
]

# --- Run the Experiment ---
print("\n--- Starting Experiment (this might take a few minutes) ---")
all_results = simulation.run_experiment(
    algorithms=algorithms_to_run,
    review_selector=review_selector,
    quantifier=quantifier,
    num_steps=NUM_SIMULATION_STEPS,
)
print("\n--- ✅ Experiment Complete ---")


# --- Evaluate and Visualize Results ---
print("\n--- Performance Metrics ---")
total_rewards = all_results.groupby('algorithm')['score'].sum()
print("Total Reward per Algorithm:")
print(total_rewards)

print("\n--- Generating Visualizations ---")
visualization.plot_cumulative_reward(
    all_results, 'Cumulative Reward: EpsilonGreedy vs. FairweatherFriend'
)
visualization.plot_restaurant_choices(
    all_results, 'Distribution of Restaurant Choices'
)
visualization.plot_average_reward_over_time(
    all_results, 'Rolling Average Reward Over Time'
)
```

---

## Interpreting the Results

After the final code cell runs, you will see three plots. Here’s what they mean:

- **Cumulative Reward**: This plot shows the total score accumulated by each algorithm over time. **A higher, steeper line is better**, meaning the algorithm consistently chose high-scoring restaurants.
- **Distribution of Restaurant Choices**: This bar chart shows how many times each restaurant was picked. A smart algorithm will **pick the best restaurants more frequently** and avoid the bad ones.
- **Rolling Average Reward**: This plot shows the average score over the last 25 choices. For a good algorithm that learns, this **line should trend upwards**, showing it's getting better at picking good restaurants over time.

---

## Customizing the Simulation

Want to experiment further? You can easily change the simulation parameters in the code block from Step 5.

- `NUM_SIMULATION_STEPS = 200`: Increase this number to run a longer simulation.
- `EPSILON = 0.1`: This controls the "EpsilonGreedy" algorithm. A value of `0.1` means it explores a random restaurant 10% of the time. Try `0.2` for more exploration or `0.05` for less.
- `model=quantify.Model.GEMINI_FLASH`: You can change the model used for scoring! Try `quantify.Model.GPT_4O` or `quantify.Model.CLAUDE_3_HAIKU`.
