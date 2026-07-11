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

        # Select cheap model: prefer a non-code model if available to avoid syntax issues on general tasks
        non_code_models = [
            m for m in self.sorted_models
            if "code" not in m.lower() and "coder" not in m.lower()
        ]
        self.cheap_model = non_code_models[0] if non_code_models else self.sorted_models[0]
        self.expensive_model = self.sorted_models[-1]

        logger.info("Default cheap model: %s", self.cheap_model)
        logger.info("Default capable model: %s", self.expensive_model)

    def _estimate_model_tier(self, model_name: str) -> float:
        """Heuristically estimates model parameter size or capability tier from its name."""
        name_lower = model_name.lower()

        # Remove version strings like v3.1, v3p1, v4.0 to avoid conflict with size matching
        name_clean = re.sub(r"v\d+(?:p|\.)\d+", "", name_lower)

        # Check for numeric parameter indicators like 8b, 70b, 405b, 8x7b, etc.
        match_b = re.search(r"(\d+x)?(\d+(?:p\d+)?)b", name_clean)
        if match_b:
            try:
                base_str = match_b.group(2)
                if "p" in base_str:
                    base = float(base_str.replace("p", "."))
                else:
                    base = float(base_str)

                if match_b.group(1):
                    multiplier = float(match_b.group(1).rstrip("x"))
                    return multiplier * base
                return base
            except ValueError:
                pass

        # Check for other numeric indicators like k2p7 (Kimi 2.7B), m3 (MiniMax 3), etc.
        match_p = re.search(r"\b[a-z]?(\d+)p(\d+)\b", name_clean)
        if match_p:
            try:
                return float(f"{match_p.group(1)}.{match_p.group(2)}")
            except ValueError:
                pass

        # Textual/Specific model brand heuristics
        if "minimax" in name_clean:
            return 300.0
        if "mixtral" in name_clean:
            return 45.0
        if "mini" in name_clean or "small" in name_clean or "lite" in name_clean:
            return 8.0
        if "medium" in name_clean:
            return 70.0
        if "large" in name_clean or "pro" in name_clean:
            return 405.0

        return 10.0  # Default fallback tier

    def get_model_for_task(self, task_type: TaskType) -> str:
        """Chooses the most cost-effective and capable model for the given task type.

        Args:
            task_type: The classified type of the task.

        Returns:
            The selected model identifier string.
        """
        # Prioritize dedicated code models for coding tasks if available
        if task_type in {TaskType.CODE_DEBUG, TaskType.CODE_GENERATION}:
            for model in self.allowed_models:
                if "code" in model.lower() or "coder" in model.lower():
                    logger.info(
                        "Selected code-specialized model '%s' for task type '%s'",
                        model,
                        task_type.value,
                    )
                    return model

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
