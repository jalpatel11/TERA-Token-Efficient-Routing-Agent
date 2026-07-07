"""Module for dynamic model selection from runtime ALLOWED_MODELS."""

from __future__ import annotations

import logging
import re
from .router import TaskType

logger = logging.getLogger(__name__)


class ModelSelector:
    """Selects the optimal model from the allowed models list for a task."""

    def __init__(self, allowed_models: list[str]) -> None:
        """Initializes the ModelSelector.

        Args:
            allowed_models: List of model identifiers available at runtime.
        """
        if not allowed_models:
            raise ValueError("ALLOWED_MODELS list cannot be empty.")

        self.allowed_models = allowed_models
        # Sort allowed models from smallest (cheapest) to largest (most capable)
        self.sorted_models = sorted(self.allowed_models, key=self._estimate_model_tier)
        logger.info(
            "Sorted allowed models by tier (smallest to largest): %s",
            self.sorted_models,
        )

        self.cheap_model = self.sorted_models[0]
        self.expensive_model = self.sorted_models[-1]

    def _estimate_model_tier(self, model_name: str) -> float:
        """Heuristically estimates model parameter size or capability tier from its name."""
        name_lower = model_name.lower()

        # Check for numeric parameter indicators like 8b, 70b, 405b, 8x7b, etc.
        match = re.search(r"(\d+x)?(\d+)b", name_lower)
        if match:
            try:
                if match.group(1):
                    multiplier = int(match.group(1).rstrip("x"))
                    base = int(match.group(2))
                    return float(multiplier * base)
                return float(match.group(2))
            except ValueError:
                pass

        # Use textual keywords as fallbacks
        if "mixtral" in name_lower:
            return 45.0
        if "mini" in name_lower or "small" in name_lower or "lite" in name_lower:
            return 8.0
        if "medium" in name_lower:
            return 70.0
        if "large" in name_lower or "pro" in name_lower:
            return 405.0

        return 10.0  # Default fallback tier

    def get_model_for_task(self, task_type: TaskType) -> str:
        """Chooses the most cost-effective and capable model for the given task type.

        Args:
            task_type: The classified type of the task.

        Returns:
            The selected model identifier string.
        """
        # Simple tasks can run on the cheaper/smaller model
        simple_tasks = {TaskType.SENTIMENT, TaskType.SUMMARY, TaskType.NER}
        if task_type in simple_tasks:
            selected = self.cheap_model
            logger.info(
                "Selected cheap model '%s' for simple task type '%s'",
                selected,
                task_type.value,
            )
            return selected

        # Complex reasoning, logic, and coding tasks need the most capable model
        selected = self.expensive_model
        logger.info(
            "Selected capable model '%s' for complex task type '%s'",
            selected,
            task_type.value,
        )
        return selected
