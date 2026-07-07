"""Client wrapper for calling the Fireworks AI API."""

from __future__ import annotations

import logging
import time
from typing import Any
import requests

logger = logging.getLogger(__name__)


class FireworksClient:
    """A client to communicate with Fireworks AI completions endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initializes the Fireworks API client.

        Args:
            api_key: The API key for authorization.
            base_url: The base URL of the Fireworks API.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts on failure.
            backoff_factor: The multiplier for exponential backoff delay.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sends a chat completion request to the Fireworks AI API with retries.

        Args:
            model: The model identifier to use.
            messages: List of message dictionaries representing conversation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            extra_params: Additional parameters for the Fireworks API request.

        Returns:
            The parsed JSON response dict.

        Raises:
            RuntimeError: If request fails after all retry attempts.
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_params:
            payload.update(extra_params)

        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                logger.info(
                    "Sending API request to model '%s' (attempt %d/%d)...",
                    model,
                    attempt,
                    self.max_retries,
                )
                response = self.session.post(
                    url, json=payload, timeout=self.timeout
                )
                latency = time.perf_counter() - start_time

                # Check for successful response
                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    logger.info(
                        "API request successful. Latency: %.3fs. Tokens: %d prompt, %d completion.",
                        latency,
                        prompt_tokens,
                        completion_tokens,
                    )
                    return data

                # Check for retriable HTTP status codes
                if response.status_code in {429, 500, 502, 503, 504}:
                    wait_time = self.backoff_factor**attempt
                    logger.warning(
                        "API request failed with status %d. Retrying in %.1fs (attempt %d/%d)...",
                        response.status_code,
                        wait_time,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "API request failed with non-retriable status %d. Response: %s",
                        response.status_code,
                        response.text,
                    )
                    response.raise_for_status()

            except requests.RequestException as exc:
                latency = time.perf_counter() - start_time
                if attempt == self.max_retries:
                    logger.error(
                        "API request encountered network error on final attempt: %s. Latency: %.3fs",
                        exc,
                        latency,
                    )
                    raise RuntimeError(
                        f"Failed to query Fireworks API after {self.max_retries} attempts: {exc}"
                    ) from exc

                wait_time = self.backoff_factor**attempt
                logger.warning(
                    "API request encountered network error: %s. Retrying in %.1fs (attempt %d/%d)...",
                    exc,
                    wait_time,
                    attempt,
                    self.max_retries,
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to query Fireworks API after {self.max_retries} attempts due to failures."
        )
