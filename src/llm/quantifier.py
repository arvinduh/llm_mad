"""Client for converting qualitative reviews into quantitative scores."""

from src import prompts

from .base import _LlmClientBase


class ReviewQuantifier(_LlmClientBase):
  """Uses an LLM to assign a numerical score to a restaurant review."""

  def quantify(self, review_text: str) -> int:
    """Scores a review by sending it to the LLM.

    Args:
        review_text: The text of the restaurant review.

    Returns:
        An integer score between 1 and 100.
    """
    prompt = prompts.QUANTIFY_REVIEW_PROMPT.format(review_text=review_text)

    try:
      response_text = self._call_api(prompt)
      score = int(response_text)
    except (ValueError, IndexError) as e:
      raise ValueError(
        f"Failed to parse valid score from LLM response: {e}"
      ) from e

    if not 1 <= score <= 100:
      raise ValueError(f"Score {score} is outside the valid range of 1-100.")

    return score
