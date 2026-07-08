"""Unit and integration tests for the TERA backend components."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from agent.fireworks_client import FireworksClient
from agent.model_selector import ModelSelector
from agent.router import TaskType
from config import Config
from main import run_pipeline


def test_config_success() -> None:
    """Tests successful loading of Configuration from environment variables."""
    env = {
        "FIREWORKS_API_KEY": "test_key",
        "FIREWORKS_BASE_URL": "https://api.fireworks.ai/inference/v1",
        "ALLOWED_MODELS": "model1, model2",
    }
    with patch.dict(os.environ, env):
        config = Config()
        assert config.api_key == "test_key"
        assert config.base_url == "https://api.fireworks.ai/inference/v1"
        assert config.allowed_models == ["model1", "model2"]


def test_config_missing_vars() -> None:
    """Tests that missing configuration variables raise ValueErrors."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Required environment variable"):
            Config()


def test_model_selector_sorting() -> None:
    """Tests model sorting by estimated size or capability."""
    allowed = [
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    ]
    selector = ModelSelector(allowed)
    assert selector.cheap_model == "accounts/fireworks/models/llama-v3p1-8b-instruct"
    assert selector.expensive_model == "accounts/fireworks/models/llama-v3p1-405b-instruct"


def test_model_selector_task_routing() -> None:
    """Tests task assignment logic (simple vs capable models)."""
    allowed = [
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
    ]
    selector = ModelSelector(allowed)

    # Simple tasks
    assert selector.get_model_for_task(TaskType.SENTIMENT) == selector.cheap_model
    assert selector.get_model_for_task(TaskType.SUMMARY) == selector.cheap_model
    assert selector.get_model_for_task(TaskType.NER) == selector.cheap_model

    # Complex tasks
    assert selector.get_model_for_task(TaskType.MATH) == selector.expensive_model
    assert selector.get_model_for_task(TaskType.CODE_DEBUG) == selector.expensive_model
    assert selector.get_model_for_task(TaskType.LOGIC) == selector.expensive_model


def test_model_selector_custom_models() -> None:
    """Tests model selector logic with the specific models provided by the user."""
    allowed = [
        "minimax-m3",
        "kimi-k2p7-code",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it-nvfp4",
    ]
    selector = ModelSelector(allowed)

    # Sorting verification
    # Only gemma models should remain:
    # 1. gemma-4-26b-a4b-it -> 26.0
    # 2. gemma-4-31b-it & nvfp4 -> 31.0
    assert selector.sorted_models[0] == "gemma-4-26b-a4b-it"
    assert "gemma-4-31b-it" in selector.sorted_models[-1]

    # Cheap model should be gemma-4-26b-a4b-it
    assert selector.cheap_model == "gemma-4-26b-a4b-it"
    # Capable model should be one of the 31b gemma models
    assert "gemma-4-31b-it" in selector.expensive_model

    # Routing logic
    # Coding tasks should fall back to the capable gemma model (no code-specific gemma model exists)
    assert "gemma-4-31b-it" in selector.get_model_for_task(TaskType.CODE_DEBUG)
    assert "gemma-4-31b-it" in selector.get_model_for_task(TaskType.CODE_GENERATION)

    # Simple tasks should route to gemma-4-26b-a4b-it
    assert selector.get_model_for_task(TaskType.SENTIMENT) == "gemma-4-26b-a4b-it"

    # Other complex tasks should route to the capable gemma model
    assert "gemma-4-31b-it" in selector.get_model_for_task(TaskType.MATH)
    assert "gemma-4-31b-it" in selector.get_model_for_task(TaskType.LOGIC)



def test_fireworks_client_success() -> None:
    """Tests successful call and response of the Fireworks client."""
    client = FireworksClient(api_key="key", base_url="https://api.test")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with patch.object(requests.Session, "post", return_value=mock_response):
        res = client.chat_completion("model", [{"role": "user", "content": "hi"}])
        assert res["choices"][0]["message"]["content"] == "Test response"


def test_fireworks_client_retries() -> None:
    """Tests that the Fireworks client retries on transient errors."""
    client = FireworksClient(
        api_key="key",
        base_url="https://api.test",
        max_retries=2,
        backoff_factor=0.1,
    )

    # Mock transient error followed by success
    mock_fail = MagicMock()
    mock_fail.status_code = 502
    mock_fail.text = "Bad Gateway"

    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.json.return_value = {
        "choices": [{"message": {"content": "Success"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }

    with patch.object(requests.Session, "post", side_effect=[mock_fail, mock_ok]) as mock_post:
        res = client.chat_completion("model", [{"role": "user", "content": "hi"}])
        assert res["choices"][0]["message"]["content"] == "Success"
        assert mock_post.call_count == 2


def test_pipeline_execution(tmp_path) -> None:
    """Integration test simulating the entire end-to-end task run."""
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [
        {"id": "t1", "prompt": "Classify sentiment: Great product!", "category": "SENTIMENT"},
        {"id": "t2", "prompt": "Solve: 2 + 2", "category": "MATH"},
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f)

    env = {
        "FIREWORKS_API_KEY": "test_key",
        "FIREWORKS_BASE_URL": "https://api.fireworks.ai/inference/v1",
        "ALLOWED_MODELS": "model-8b,model-70b",
        "INPUT_FILE": str(input_file),
        "OUTPUT_FILE": str(output_file),
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Positive: Great product!\nAnswer: 4"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }

    with patch.dict(os.environ, env), \
         patch.object(requests.Session, "post", return_value=mock_resp), \
         pytest.raises(SystemExit) as excinfo:
        run_pipeline()

    # The pipeline should complete and call sys.exit(0)
    assert excinfo.value.code == 0

    assert os.path.exists(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert len(results) == 2
    assert results[0]["id"] == "t1"
    assert results[1]["id"] == "t2"
