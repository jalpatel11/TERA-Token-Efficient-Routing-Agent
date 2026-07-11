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
    # Sorted order should be:
    # 1. kimi-k2p7-code (2.7)
    # 2. gemma-4-26b-a4b-it (26.0)
    # 3. gemma-4-31b-it (31.0)
    # 4. gemma-4-31b-it-nvfp4 (31.0)
    # 5. minimax-m3 (300.0)
    assert selector.sorted_models[0] == "kimi-k2p7-code"
    assert selector.sorted_models[-1] == "minimax-m3"

    # Cheap model should be gemma-4-26b-a4b-it (non-code, smallest)
    assert selector.cheap_model == "gemma-4-26b-a4b-it"
    # Capable model should be minimax-m3
    assert selector.expensive_model == "minimax-m3"

    # Routing logic
    # Coding tasks should route to kimi-k2p7-code
    assert selector.get_model_for_task(TaskType.CODE_DEBUG) == "kimi-k2p7-code"
    assert selector.get_model_for_task(TaskType.CODE_GENERATION) == "kimi-k2p7-code"

    # Simple tasks should route to cheap model (gemma-4-26b-a4b-it)
    assert selector.get_model_for_task(TaskType.SENTIMENT) == "gemma-4-26b-a4b-it"

    # Other complex tasks should route to capable model (minimax-m3)
    assert selector.get_model_for_task(TaskType.MATH) == "minimax-m3"
    assert selector.get_model_for_task(TaskType.LOGIC) == "minimax-m3"



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
        {"task_id": "t1", "prompt": "Classify sentiment: Great product!", "category": "SENTIMENT"},
        {"task_id": "t2", "prompt": "Solve: 2 + 2", "category": "MATH"},
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
    assert results[0]["task_id"] == "t1"
    assert results[1]["task_id"] == "t2"


def test_gemma_client_generate_answer() -> None:
    """Tests the generate_answer method of GemmaClient with a mock LLM."""
    from agent.gemma_client import GemmaClient

    with patch("agent.gemma_client.Llama") as mock_llama_class:
        mock_llama_instance = MagicMock()
        mock_llama_instance.return_value = {
            "choices": [{"text": "Positive: It's good"}]
        }
        mock_llama_class.return_value = mock_llama_instance

        client = GemmaClient(model_path="dummy.gguf")
        client.llm = mock_llama_instance

        ans = client.generate_answer("Good stuff", "SENTIMENT")
        assert ans == "Positive: It's good"
        assert mock_llama_instance.called


@patch("main.GemmaClient")
def test_pipeline_time_budgeted_routing_local(mock_gemma_client_class, tmp_path) -> None:
    """Tests that a simple task is executed locally when time budget is sufficient."""
    mock_gemma_instance = MagicMock()
    mock_gemma_instance.is_available.return_value = True
    mock_gemma_instance.generate_answer.return_value = "Positive: Great product!"
    mock_gemma_client_class.return_value = mock_gemma_instance

    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [
        {"task_id": "t1", "prompt": "Classify sentiment: Great product!", "category": "SENTIMENT"},
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f)

    env = {
        "FIREWORKS_API_KEY": "test_key",
        "FIREWORKS_BASE_URL": "https://api.test",
        "ALLOWED_MODELS": "cheap,expensive",
        "INPUT_FILE": str(input_file),
        "OUTPUT_FILE": str(output_file),
    }

    # Mock time.perf_counter to return 0.0, so remaining time = 58s, required time = 3.5s
    with patch.dict(os.environ, env), \
         patch("time.perf_counter", return_value=0.0), \
         pytest.raises(SystemExit) as excinfo:
        run_pipeline()

    assert excinfo.value.code == 0
    assert os.path.exists(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    assert len(results) == 1
    assert results[0]["task_id"] == "t1"
    assert "Positive" in results[0]["answer"]
    mock_gemma_instance.generate_answer.assert_called_once()


@patch("main.GemmaClient")
def test_pipeline_time_budgeted_routing_remote_fallback(mock_gemma_client_class, tmp_path) -> None:
    """Tests that a simple task falls back to remote when time budget is low."""
    mock_gemma_instance = MagicMock()
    mock_gemma_instance.is_available.return_value = True
    mock_gemma_client_class.return_value = mock_gemma_instance

    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [
        {"task_id": "t1", "prompt": "Classify sentiment: Great product!", "category": "SENTIMENT"},
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f)

    env = {
        "FIREWORKS_API_KEY": "test_key",
        "FIREWORKS_BASE_URL": "https://api.test",
        "ALLOWED_MODELS": "cheap,expensive",
        "INPUT_FILE": str(input_file),
        "OUTPUT_FILE": str(output_file),
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Positive: Great!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    # First call returns 0.0 (start_time), all subsequent calls return 56.0
    calls = []
    def dynamic_counter():
        if not calls:
            calls.append(1)
            return 0.0
        return 56.0

    with patch.dict(os.environ, env), \
         patch("time.perf_counter", side_effect=dynamic_counter), \
         patch.object(requests.Session, "post", return_value=mock_resp) as mock_post, \
         pytest.raises(SystemExit) as excinfo:
        run_pipeline()

    assert excinfo.value.code == 0
    mock_gemma_instance.generate_answer.assert_not_called()
    assert mock_post.call_count == 1


