"""Stores prompts for interacting with language models."""

QUANTIFY_REVIEW_PROMPT = """
Analyze the following restaurant review and assign it a quantitative score from 1 to 100.
A score of 1 represents an extremely negative experience, while a score of 100 represents an overwhelmingly positive one.
Consider the tone, specific details about food quality, service, and ambiance.

The final output must be a single integer number between 1 and 100 and nothing else. Do not add any explanation or surrounding text.

Review:
"{review_text}"

Score:
"""

CLASSIFY_REVIEW_PROMPT = """
Is the following restaurant review "Good" or "Bad"?
Consider the tone, specific details about food quality, service, and ambiance.

The final output must be the single word "Good" or the single word "Bad" and nothing else.

Review:
"{review_text}"

Classification:
"""
