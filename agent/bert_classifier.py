"""Local inference helper using the fine-tuned DistilBERT difficulty classifier."""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

from .router import TaskType

logger = logging.getLogger(__name__)

# Global singleton elements for lazy loading
_TOKENIZER = None
_MODEL = None
_ATTEMPTED_LOAD = False

MODEL_DIR: Final[str] = os.environ.get(
    "LOCAL_ROUTER_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "router")
)


def _load_model() -> bool:
    """Lazy loads the DistilBERT tokenizer and model from local path.

    Returns:
        True if loaded successfully, False otherwise.
    """
    global _TOKENIZER, _MODEL, _ATTEMPTED_LOAD
    if _ATTEMPTED_LOAD:
        return _MODEL is not None

    _ATTEMPTED_LOAD = True

    # Validate that config exists in target path
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(config_path):
        logger.warning(
            "DistilBERT router weights config not found at '%s'. "
            "TERA will run in fallback category-based routing mode.",
            config_path
        )
        return False

    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        
        logger.info("Loading local DistilBERT sequence classifier from '%s'...", MODEL_DIR)
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        _MODEL = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        _MODEL.eval()
        logger.info("Local DistilBERT sequence classifier loaded successfully.")
        return True
    except Exception as exc:
        logger.error("Failed to load local DistilBERT sequence classifier: %s", exc)
        return False


def predict_difficulty(prompt: str, task_type: TaskType | None = None) -> str:
    """Predicts whether the prompt is EASY or HARD for local execution.

    Args:
        prompt: The user task prompt text.
        task_type: Optional pre-classified category of the task.

    Returns:
        "easy" or "hard" based on DistilBERT prediction or category fallback.
    """
    if not prompt:
        return "easy"

    # Attempt to load and run DistilBERT classifier
    if _load_model() and _MODEL is not None and _TOKENIZER is not None:
        try:
            import torch
            with torch.no_grad():
                inputs = _TOKENIZER(
                    prompt,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )
                outputs = _MODEL(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1).item()
                # 0 = easy, 1 = hard
                result = "easy" if predictions == 0 else "hard"
                logger.info("DistilBERT model predicted difficulty: '%s' for prompt.", result)
                return result
        except Exception as exc:
            logger.error("DistilBERT inference failed: %s. Falling back to category rule.", exc)
            # Proceed to fallback below

    # Fallback Rule:
    if task_type is None:
        # Analyze prompt directly for fallback
        normalized = prompt.lower()
        hard_keywords = (
            "puzzle", "logic", "deduce", "infer", "calculate", "equation", "solve",
            "math", "sum", "debug", "bug", "traceback", "implement", "code", "python"
        )
        is_hard = any(kw in normalized for kw in hard_keywords)
        fallback_result = "hard" if is_hard else "easy"
    else:
        # Sentiment, Summary, NER, General are easy fallback.
        # Math, Logic, Code Gen, Code Debug are hard fallback.
        is_simple = task_type in {
            TaskType.SENTIMENT,
            TaskType.SUMMARY,
            TaskType.NER,
            TaskType.GENERAL
        }
        fallback_result = "easy" if is_simple else "hard"

    logger.info(
        "Using fallback difficulty: '%s' (task_type: %s).",
        fallback_result,
        task_type.value if task_type else "None"
    )
    return fallback_result
