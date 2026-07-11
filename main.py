"""Main entry point for the TERA routing agent execution pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from agent.formatter import format_response
from agent.prompts import build_prompt
from agent.router import normalize_task_type, route, TaskType
from agent.fireworks_client import FireworksClient
from agent.model_selector import ModelSelector
from agent.gemma_client import GemmaClient
from agent.bert_classifier import predict_difficulty
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

    # Initialize client, model selector, and local Gemma client
    client = FireworksClient(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    model_selector = ModelSelector(allowed_models=config.allowed_models)
    gemma_client = GemmaClient()

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
    
    total_tasks = len(tasks)
    total_local_tasks = 0
    total_cheap_remote_tasks = 0
    total_expensive_remote_tasks = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            logger.warning("Skipping invalid task at index %d (not a dictionary).", index)
            continue

        # Extract task properties with sensible fallbacks
        task_id = task.get("task_id") or task.get("id") or f"task-{index}"
        prompt = task.get("prompt") or task.get("input") or task.get("text") or task.get("question")
        requested_format = task.get("requested_format") or task.get("format")

        if not prompt:
            logger.error("Task '%s' has no valid prompt. Skipping.", task_id)
            continue

        logger.info("Processing task '%s'...", task_id)
        task_start = time.perf_counter()

        try:
            # Step 1: Difficulty Classification (using DistilBERT classifier directly first)
            difficulty = predict_difficulty(prompt)
            is_easy_task = (difficulty == "easy")

            # Step 2: Routing (task classification)
            task_type = route(prompt, gemma_client=gemma_client)
            logger.info(
                "Routed task '%s' to category '%s' using local classifier/heuristics. Difficulty: %s",
                task_id, task_type.value, difficulty
            )

            # Dynamic time budget calculation
            remaining_tasks = total_tasks - index
            elapsed_time = time.perf_counter() - start_time
            remaining_time = 58.0 - elapsed_time  # 60s total limit, with 2s safety buffer
            time_required_for_local = 3.5 * remaining_tasks

            if is_easy_task and gemma_client.is_available() and remaining_time >= time_required_for_local:
                # --- TIER 1: Local Gemma ---
                logger.info(
                    "Executing task '%s' LOCALLY using Gemma (Tier 1). Remaining time: %.1fs, Required: %.1fs (0 Fireworks tokens)",
                    task_id, remaining_time, time_required_for_local
                )
                raw_output = gemma_client.generate_answer(prompt, task_type)
                formatted_output = format_response(
                    task_type=task_type,
                    output=raw_output,
                    requested_format=requested_format,
                )
                task_latency = time.perf_counter() - task_start
                logger.info(
                    "Task '%s' completed locally in %.3fs. Model: local-gemma. Tokens: 0 prompt, 0 completion.",
                    task_id, task_latency
                )
                total_local_tasks += 1
            else:
                # Remote execution (Tier 2 or Tier 3)
                # Step 3: Model Selection based on predicted difficulty
                if is_easy_task:
                    model = model_selector.cheap_model
                else:
                    model = model_selector.get_model_for_task(task_type)
                
                is_cheap_model = (model == model_selector.cheap_model)
                if is_cheap_model:
                    # --- TIER 2: Cheap Remote ---
                    logger.info(
                        "Routing task '%s' to CHEAP remote model '%s' (Tier 2).",
                        task_id, model
                    )
                    total_cheap_remote_tasks += 1
                else:
                    # --- TIER 3: Capable Remote ---
                    logger.info(
                        "Routing task '%s' to CAPABLE remote model '%s' (Tier 3).",
                        task_id, model
                    )
                    total_expensive_remote_tasks += 1

                # Step 4: Local reasoning hints (only for Math/Logic tasks to save remote completion tokens)
                # Ensure we have enough time budget for reasoning hints (~1.5s per task)
                hints = None
                if (task_type in {TaskType.MATH, TaskType.LOGIC} and 
                        gemma_client.is_available() and 
                        (remaining_time >= time_required_for_local + 1.5)):
                    logger.info("Generating local reasoning hints using Gemma for task '%s'...", task_id)
                    hints = gemma_client.generate_reasoning_hints(prompt)

                # Step 5: Prompt Building (incorporating local hints if available)
                prompt_str = build_prompt(prompt, task_type, hints=hints)

                # Step 5: Call API Client
                messages = [{"role": "user", "content": prompt_str}]
                response_data = client.chat_completion(
                    model=model,
                    messages=messages,
                )

                choices = response_data.get("choices", [])
                if not choices:
                    raise RuntimeError("API returned response with no choices.")

                raw_output = choices[0].get("message", {}).get("content", "")

                # Step 6: Formatting Response
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

                logger.info(
                    "Task '%s' completed remotely in %.3fs. Model: %s. Tokens: %d prompt, %d completion.",
                    task_id, task_latency, model, prompt_tokens, completion_tokens
                )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            results.append({
                "task_id": task_id,
                "answer": formatted_output
            })

        except Exception as exc:
            logger.error("Error processing task '%s': %s", task_id, exc)
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
    logger.info(
        "Execution Summary:\n"
        "  Total Tasks Processed: %d\n"
        "  - Routed to Local Gemma (Tier 1): %d\n"
        "  - Routed to Cheap Remote (Tier 2): %d\n"
        "  - Routed to Capable Remote (Tier 3): %d\n"
        "  Total Tokens Consumed:\n"
        "    Prompt Tokens:     %d\n"
        "    Completion Tokens: %d\n"
        "    Total Tokens:      %d",
        total_tasks,
        total_local_tasks,
        total_cheap_remote_tasks,
        total_expensive_remote_tasks,
        total_prompt_tokens,
        total_completion_tokens,
        total_prompt_tokens + total_completion_tokens
    )
    logger.info("Pipeline completed successfully in %.3fs.", total_latency)
    sys.exit(0)


if __name__ == "__main__":
    run_pipeline()
