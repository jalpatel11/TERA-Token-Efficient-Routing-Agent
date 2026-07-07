"""Main entry point for the TERA routing agent execution pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from agent.formatter import format_response
from agent.prompts import build_prompt
from agent.router import normalize_task_type, route
from agent.fireworks_client import FireworksClient
from agent.model_selector import ModelSelector
from config import Config

# Configure logging to standard output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tera_pipeline")


def run_pipeline() -> None:
    """Executes the end-to-end task routing and processing pipeline."""
    logger.info("Initializing TERA pipeline...")
    start_time = time.perf_counter()

    # Load configuration
    try:
        config = Config()
    except Exception as exc:
        logger.critical("Configuration initialization failed: %s", exc)
        sys.exit(1)

    # Initialize client and selector
    client = FireworksClient(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    model_selector = ModelSelector(allowed_models=config.allowed_models)

    # Determine input and output file paths (allowing env overrides for local testing)
    input_path = os.environ.get("INPUT_FILE", "/input/tasks.json")
    output_path = os.environ.get("OUTPUT_FILE", "/output/results.json")

    logger.info("Reading input tasks from '%s'...", input_path)
    if not os.path.exists(input_path):
        logger.critical("Input file '%s' does not exist.", input_path)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as exc:
        logger.critical("Failed to parse input JSON file '%s': %s", input_path, exc)
        sys.exit(1)

    if not isinstance(tasks, list):
        logger.critical("Input tasks JSON must be a list of task objects.")
        sys.exit(1)

    logger.info("Loaded %d tasks to process.", len(tasks))
    results = []

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            logger.warning("Skipping invalid task at index %d (not a dictionary).", index)
            continue

        # Extract task properties with sensible fallbacks
        task_id = task.get("id") or task.get("task_id") or f"task-{index}"
        prompt = task.get("prompt") or task.get("input") or task.get("text") or task.get("question")
        requested_format = task.get("requested_format") or task.get("format")

        if not prompt:
            logger.error("Task '%s' has no valid prompt. Skipping.", task_id)
            continue

        logger.info("Processing task '%s'...", task_id)
        task_start = time.perf_counter()

        try:
            # Step 1: Routing (task classification)
            raw_category = task.get("task_type") or task.get("category")
            if raw_category:
                task_type = normalize_task_type(raw_category)
                logger.info(
                    "Task '%s' pre-classified as '%s'. Normalized to '%s'.",
                    task_id, raw_category, task_type.value
                )
            else:
                task_type = route(prompt)
                logger.info(
                    "Routed task '%s' to category '%s' using heuristics.",
                    task_id, task_type.value
                )

            # Step 2: Model Selection
            model = model_selector.get_model_for_task(task_type)

            # Step 3: Prompt Building
            prompt_str = build_prompt(prompt, task_type)

            # Step 4: Call API Client
            messages = [{"role": "user", "content": prompt_str}]
            response_data = client.chat_completion(
                model=model,
                messages=messages,
            )

            choices = response_data.get("choices", [])
            if not choices:
                raise RuntimeError("API returned response with no choices.")

            raw_output = choices[0].get("message", {}).get("content", "")

            # Step 5: Formatting Response
            formatted_output = format_response(
                task_type=task_type,
                output=raw_output,
                requested_format=requested_format,
            )

            # Record task metrics
            task_latency = time.perf_counter() - task_start
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Local execution log (not written to results.json)
            logger.info(
                "Task '%s' completed in %.3fs. Model: %s. Tokens: %d prompt, %d completion.",
                task_id, task_latency, model, prompt_tokens, completion_tokens
            )

            results.append({
                "id": task_id,
                "result": formatted_output
            })

        except Exception as exc:
            logger.error("Error processing task '%s': %s", task_id, exc)
            # For unrecoverable execution failure, we terminate the process
            sys.exit(1)

    # Step 6: Write outputs
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as exc:
            logger.critical("Failed to create output directory '%s': %s", output_dir, exc)
            sys.exit(1)

    logger.info("Writing results to '%s'...", output_path)
    try:
        # Validate that we are writing valid JSON
        json_output = json.dumps(results, indent=2, ensure_ascii=False)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_output)
    except Exception as exc:
        logger.critical("Failed to write results JSON to '%s': %s", output_path, exc)
        sys.exit(1)

    total_latency = time.perf_counter() - start_time
    logger.info("Pipeline completed successfully in %.3fs.", total_latency)
    sys.exit(0)


if __name__ == "__main__":
    run_pipeline()
