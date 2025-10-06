"""Converts qualitative reviews into quantitative scores using an LLM."""

import json
import random
import time
from enum import Enum

import requests

from src import prompts


class Model(str, Enum):
  """Enumeration of supported models on OpenRouter."""

  GEMINI_FLASH = "google/gemini-flash"
  GEMINI_PRO = "google/gemini-pro"
  GPT_4O = "openai/gpt-4o"
  CLAUDE_3_HAIKU = "anthropic/claude-3-haiku-20240307"


class ReviewQuantifier:
  """Uses an LLM to assign a numerical score to a restaurant review."""

  _API_URL = "https://openrouter.ai/api/v1/chat/completions"
  _MAX_RETRIES = 5
  _INITIAL_BACKOFF = 1.0  # In seconds

  def __init__(
    self,
    api_key: str,
    model: Model = Model.GEMINI_FLASH,
    site_url: str = "https://github.com/arvinduh/llm_mad",
    app_name: str = "LLM_MAD_Project",
  ) -> None:
    """Initializes the ReviewQuantifier."""
    if not api_key:
      raise ValueError("API key cannot be empty.")
    self._model = model.value
    self._headers = {
      "Authorization": f"Bearer {api_key}",
      "HTTP-Referer": site_url,
      "X-Title": app_name,
      "Content-Type": "application/json",
    }

  def quantify(self, review_text: str) -> int:
    """Scores a review, handling retries and rate limiting."""
    prompt = prompts.QUANTIFY_REVIEW_PROMPT.format(review_text=review_text)
    payload = {
      "model": self._model,
      "messages": [{"role": "user", "content": prompt}],
    }
    backoff_time = self._INITIAL_BACKOFF

    for attempt in range(self._MAX_RETRIES):
      try:
        response = requests.post(
          self._API_URL,
          headers=self._headers,
          data=json.dumps(payload),
          timeout=30,
        )
        if response.status_code == 429:  # Rate limited
          print(f"Rate limited. Retrying in {backoff_time:.2f} seconds...")
          time.sleep(backoff_time)
          # Exponential backoff with jitter
          backoff_time *= 2 * (1 + random.random())
          continue

        response.raise_for_status()  # Raises HTTPError for other bad responses
        response_json = response.json()

        content = response_json["choices"][0]["message"]["content"]
        score = int(content.strip())

        if not 1 <= score <= 100:
          raise ValueError(f"Score {score} is outside valid range 1-100.")

        return score

      except requests.exceptions.RequestException as e:
        if attempt == self._MAX_RETRIES - 1:
          raise RuntimeError(
            f"API request failed after {self._MAX_RETRIES} attempts: {e}"
          ) from e
        time.sleep(backoff_time)  # Wait before retrying on connection errors
      except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        raise ValueError(
          f"Failed to parse valid score from LLM response: {e}"
        ) from e

    raise RuntimeError(
      f"Failed to get a valid response after {self._MAX_RETRIES} retries."
    )
