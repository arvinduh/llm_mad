"""Defines a base class for LLM API clients."""

import abc
import json
import random
import time
from enum import Enum

import requests


class Model(str, Enum):
  """Enumeration of supported models on OpenRouter."""

  GEMINI_FLASH = "google/gemini-flash"
  GEMINI_PRO = "google/gemini-pro"
  GPT_4O = "openai/gpt-4o"
  CLAUDE_3_HAIKU = "anthropic/claude-3-haiku-20240307"


class _LlmClientBase(abc.ABC):
  """Abstract base class for clients that call an LLM API."""

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
    """Initializes the client."""
    if not api_key:
      raise ValueError("API key cannot be empty.")
    self._model = model.value
    self._headers = {
      "Authorization": f"Bearer {api_key}",
      "HTTP-Referer": site_url,
      "X-Title": app_name,
      "Content-Type": "application/json",
    }

  def _call_api(self, prompt: str) -> str:
    """Handles the core logic of calling the LLM API with retries."""
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
          backoff_time *= 2 * (1 + random.random())
          continue

        response.raise_for_status()
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"].strip()

      except requests.exceptions.RequestException as e:
        if attempt == self._MAX_RETRIES - 1:
          raise RuntimeError(
            f"API request failed after {self._MAX_RETRIES} attempts: {e}"
          ) from e
        time.sleep(backoff_time)

    raise RuntimeError(
      f"Failed to get a valid response after {self._MAX_RETRIES} retries."
    )
