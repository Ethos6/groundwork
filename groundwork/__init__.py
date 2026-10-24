"""Groundwork: make an existing LLM better at doing tasks, more grounded, and on-tone.

Two halves that can be used independently:

* ``groundwork.runtime`` -- an inference-time layer (no training): tool-using agent,
  hallucination guard, and tone enforcement around any chat model.
* ``groundwork.data`` + ``groundwork.train`` -- a fine-tuning kit that builds verified
  training data (task traces, abstention examples, tone examples) and fine-tunes the model.
"""

__version__ = "0.1.0"
