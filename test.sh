#!/bin/bash

# Exit on any error
set -e

# Change directory to the repository root
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$CDIR"

# Load environment variables from .env if it exists, otherwise fallback to .env.example
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
elif [ -f .env.example ]; then
    echo "Warning: .env not found. Loading environment variables from .env.example..."
    export $(grep -v '^#' .env.example | xargs)
else
    echo "Error: Neither .env nor .env.example exists."
    exit 1
fi

# Ensure output directory exists
mkdir -p output

# Set local paths for test runs
export INPUT_FILE=${INPUT_FILE:-input/tasks.json}
export OUTPUT_FILE=${OUTPUT_FILE:-output/results.json}

# Run pipeline and capture log output to both console and a temp log file
LOG_FILE="output/test_run.log"
echo "Starting TERA pipeline execution..."
echo "------------------------------------------------"
.venv/bin/python main.py 2>&1 | tee "$LOG_FILE"
echo "------------------------------------------------"

# Parse token metrics from the log file
PROMPT_TOKENS=$(grep -a "tera_pipeline:" "$LOG_FILE" | grep -a -oE "Tokens: [0-9]+ prompt" | awk '{sum+=$2} END {print sum}')
COMPLETION_TOKENS=$(grep -a "tera_pipeline:" "$LOG_FILE" | grep -a -oE "[0-9]+ completion" | awk '{sum+=$1} END {print sum}')

PROMPT_TOKENS=${PROMPT_TOKENS:-0}
COMPLETION_TOKENS=${COMPLETION_TOKENS:-0}
TOTAL_TOKENS=$((PROMPT_TOKENS + COMPLETION_TOKENS))

echo "Execution Summary:"
echo "  Prompt Tokens:     $PROMPT_TOKENS"
echo "  Completion Tokens: $COMPLETION_TOKENS"
echo "  Total Tokens:      $TOTAL_TOKENS"
