# LLM-Powered Multi-Armed Bandits for Restaurant Reviews

This project implements and evaluates multi-armed bandit algorithms that use a
Large Language Model (LLM) to quantify qualitative restaurant reviews. It
compares a baseline "Fairweather Friend" algorithm with a proposed
"Epsilon-Greedy on Steroids" algorithm.

[TOC]

## Project Structure

- `llm_mad/`: The core Python package.
  - `models/`: Contains the bandit algorithm implementations.
  - `quantify.py`: Handles scoring reviews using an LLM via the OpenRouter API.
  - `reviews.py`: Manages the dataset of restaurant reviews.
  - `prompts.py`: Stores the LLM prompt for review quantification.
- `data/`: Contains the raw `reviews.csv` dataset.
- `setup.py`: Makes the project installable as a package.
- `requirements.txt`: Project dependencies.

## Setup in Google Colab

1.  **Clone the Repository**

    ```bash
    !git clone <YOUR_REPOSITORY_URL> llm_mad
    %cd llm_mad
    ```

2.  **Install Dependencies**

    This command uses the `setup.py` file to install your project in "editable"
    mode, making the `llm_mad` package importable.

    ```bash
    !pip install -e . -r requirements.txt
    ```

3.  **Set API Key**

    You will need an API key from [OpenRouter](https://openrouter.ai/). Store it
    using Colab's secret manager.

    ```python
    from google.colab import userdata

    # Access your secret
    # Ensure you have a secret named 'OPENROUTER_API_KEY' in your Colab notebook
    OPENROUTER_API_KEY = userdata.get('OPENROUTER_API_KEY')
    ```

## Usage

After setup, you can import and use the modules directly in your notebook.

```python
from llm_mad import models
from llm_mad import quantify
from llm_mad import reviews

# Initialize the Epsilon-Greedy model
epsilon_model = models.EpsilonGreedy(
    restaurants=['Restaurant A', 'Restaurant B'],
    epsilon=0.1
)

# Use the model
choice = epsilon_model.select_restaurant()
print(f"Selected: {choice}")
```
