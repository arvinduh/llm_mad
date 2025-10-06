"""Client for converting qualitative reviews into binary classifications."""

from src import prompts

from .base import _LlmClientBase


class ReviewClassifier(_LlmClientBase):
  """Uses an LLM to classify a restaurant review as 'Good' or 'Bad'."""

  def classify(self, review_text: str) -> str:
    """Classifies a review as 'Good' or 'Bad'.

    Args:
        review_text: The text of the restaurant review.

    Returns:
        The string "Good" or "Bad".
    """
    prompt = prompts.CLASSIFY_REVIEW_PROMPT.format(review_text=review_text)
    response_text = self._call_api(prompt)

    if response_text not in ("Good", "Bad"):
      raise ValueError(f'Invalid classification returned: "{response_text}"')

    return response_text
