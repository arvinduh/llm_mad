"""Defines a base class for LLM API clients."""

import abc
import json
import random
import time
from enum import Enum
from typing import List, Union

import requests


class Model(str, Enum):
  """Enumeration of supported models on OpenRouter."""

  # Note: IDs are corrected based on OpenRouter's current naming.
  # Gemini Flash can sometimes be unavailable, so having fallbacks is good.
  GEMINI_FLASH = "google/gemini-flash-1.5"
  CLAUDE_3_HAIKU = "anthropic/claude-3-haiku"
  GEMINI_PRO = "google/gemini-pro-1.5"
  GPT_4O = "openai/gpt-4o"


class _LlmClientBase(abc.ABC):
  """Abstract base class for clients that call an LLM API."""

  _API_URL = "https://openrouter.ai/api/v1/chat/completions"
  _MAX_RETRIES = 5
  _INITIAL_BACKOFF = 1.0  # In seconds

  def __init__(
    self,
    api_key: str,
    # Allow a single model or a list of models for fallback.
    model: Union[Model, List[Model]] = Model.GEMINI_FLASH,
    site_url: str = "https://github.com/arvinduh/llm_mad",
    app_name: str = "LLM_MAD_Project",
  ) -> None:
    """Initializes the client."""
    if not api_key:
      raise ValueError("API key cannot be empty.")

    # Store a list of models to try. If only one is provided, make it a list.
    if isinstance(model, Model):
      self._models = [model.value]
    else:
      self._models = [m.value for m in model]

    # Use standard header names; some proxies/hosts reject nonstandard keys.
    self._headers = {
      "Authorization": f"Bearer {api_key}",
      "Referer": site_url,
      "X-App-Name": app_name,
      "Content-Type": "application/json",
    }

  def _call_api(self, prompt: str) -> str:
    """Handles the core logic of calling the LLM API with retries and fallbacks."""
    last_error = "No models were provided to attempt."

    # Outer loop to iterate through the list of models (for fallback).
    for model_name in self._models:
      print(f"Attempting to use model: {model_name}")
      payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
      }
      backoff_time = self._INITIAL_BACKOFF

      # Inner loop for retrying a single model on transient errors.
      for attempt in range(self._MAX_RETRIES):
        try:
          response = requests.post(
            self._API_URL,
            headers=self._headers,
            data=json.dumps(payload),
            timeout=30,
          )

          # --- Start of Response Handling ---

          # On success, parse and return the content immediately.
          if response.ok:
            response_json = response.json()
            try:
              content = response_json["choices"][0]["message"][
                "content"
              ].strip()
              return content
            except (KeyError, IndexError, TypeError) as e:
              raise RuntimeError(
                f"Unexpected API response shape: {response_json}"
              ) from e

          # Handle specific, non-successful status codes.
          last_error_body = response.text or "<unable to read response body>"
          last_error = f"{response.status_code}: {last_error_body}"

          # 429: Rate limited. Wait and retry this same model.
          if response.status_code == 429:
            print(f"Rate limited. Retrying in {backoff_time:.2f} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2 * (1 + random.random())
            continue

          # 404: Model not found or unavailable. Stop retrying and move to the next model.
          if response.status_code == 404:
            print(
              f"Warning: Model '{model_name}' not found or unavailable (404)."
            )
            break  # Breaks from the retry loop to try the next model in the outer loop.

          # For other client/server errors (e.g., 400, 500), fail fast.
          raise RuntimeError(
            f"LLM API returned unrecoverable error: {last_error}"
          )

        except requests.exceptions.RequestException as e:
          last_error = str(e)
          if attempt == self._MAX_RETRIES - 1:
            print(f"Network error with model '{model_name}': {e}")
            break  # Stop retrying this model and move to the next one.
          time.sleep(backoff_time)

      # This line is reached if the retry loop for a model completes without success or is broken.
      # The outer loop will then proceed to the next model.

    # If the outer loop completes without returning a successful response, all models have failed.
    raise RuntimeError(
      f"Failed to get a valid response from any of the specified models. Last error: {last_error}"
    )
