# TERA: Token Efficient Routing Agent

TERA is a Hybrid Token-Efficient Routing Agent built for the AMD ACT II Hackathon (Track 1). It dynamically routes prompts using a three-tier architecture to minimize token consumption while maintaining output accuracy and protecting against container execution timeouts.

---

## How It Works

The execution pipeline runs in 6 distinct phases:

1. **Centralized Configuration (`config.py`)**
   Loads and validates environment variables (`FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`) at startup. Enforces strict presence checks.

2. **Model Capability Ranking & Selection (`agent/model_selector.py`)**
   Inspects names in `ALLOWED_MODELS` and ranks them from smallest (cheapest) to largest (most capable) using parameter-size heuristic parsing (e.g. matching `26b`, `k2p7`, `m3`, etc.).
   * **Code-Specialized Priority**: If a task requires code generation or debugging, the selector checks for and routes to a model containing `code` or `coder` in its name (e.g. `kimi-k2p7-code`).
   * **General Selection**: Resolves the optimal cheap fallback model (`gemma-4-26b-a4b-it`) and flagship reasoning model (`minimax-m3`) dynamically.

3. **Dynamic Time-Budgeted Three-Tier Routing (`main.py` & `agent/router.py`)**
   Classifies tasks into 8 required categories using local keyword heuristics. To maximize token efficiency, it dynamically manages execution time against the 60-second limit:
   * **Tier 1 (Local Gemma - 0 Tokens)**: Simple/moderate tasks (Sentiment, NER, Summarization, General) are resolved locally on the pre-baked Gemma model, incurring **0 Fireworks tokens**, provided the remaining time budget is sufficient.
   * **Tier 2 (Cheap Remote - Low Tokens)**: Fallback to the cheap remote model (`gemma-4-26b-a4b-it`) if the time budget runs low, avoiding container timeouts.
   * **Tier 3 (Capable Remote - Higher Tokens)**: Hard math, logic, and code tasks route directly to flagship remote models (`minimax-m3` or `kimi-k2p7-code`) for maximum accuracy.

4. **API Execution (`agent/fireworks_client.py`)**
   Wraps HTTP POST requests to the Fireworks completions endpoint.
   * **Context Compression**: Integrates the `headroom` library to intelligently compress input prompt contexts to minimize remote prompt tokens.
   * **Robust Retries**: Handles transient network and API errors (HTTP 429, 502, 503, 504) using exponential backoff retry cycles.

5. **Deterministic Formatting (`agent/formatter.py`)**
   Normalizes output responses depending on task types (extracting JSON entities for NER, stripping markdown code fences, and normalizing math expressions).

6. **Log & Export Output (`main.py`)**
   Calculates pipeline execution stats, logs token metrics, and writes the formatted predictions to the output file.

---

## How to Configure and Run the Code

### 1. Environment Settings
Create a `.env` file in the root directory (based on `.env.example`).
```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code
```
*(Note: `.env` is ignored by Git and Docker to prevent credentials from leaking into repository check-ins or built images).*

### 2. Run Local Evaluation Pipeline
To test the pipeline on the host machine using a Python virtual environment:
1. Ensure your python virtual environment is initialized:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. Run the automated test script:
   ```bash
   ./test.sh
   ```
   This will run `main.py` against the tasks listed in `input/tasks.json`, write results to `output/results.json`, and print a token consumption summary.

### 3. Run unit tests
To run the automated pytest test suite:
```bash
.venv/bin/pytest
```

### 4. Build and Run Containerized Pipeline
To package the agent into a lightweight or standard GGUF-bundled Docker container:
1. Build the image:
   ```bash
   docker build -t tera-agent .
   ```
2. Run the container:
   ```bash
   docker run --rm \
     -v $(pwd)/input:/input \
     -v $(pwd)/output:/output \
     -e FIREWORKS_API_KEY="your_api_key" \
     -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
     -e ALLOWED_MODELS="accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code" \
     tera-agent
   ```

---

## Current Version Benchmark Results

We evaluated TERA locally on a mock dataset containing one task for each of the 8 capability categories.

### System Configuration used for Test Run:
* **Models**: `gemma-4-26b-a4b-it` (Cheap fallback), `minimax-m3` (Capable logic/math), `kimi-k2p7-code` (Capable code)
* **Local Model**: `google_gemma-4-E4B-it-Q4_K_M.gguf` (Tier 1 Local)
* **Total Elapsed Latency**: 34.4s (~4.3s average per task)

### Dynamic Routing Details:

| Task ID | Task Category | Routing Tier / Model | Status | Fireworks Prompt Tokens | Fireworks Completion Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `t-general` | General | **Tier 1 (Local Gemma)** | Success | 0 | 0 |
| `t-math` | Math | **Tier 3 (minimax-m3)** | Success | 360 | 109 |
| `t-sentiment` | Sentiment | **Tier 1 (Local Gemma)** | Success | 0 | 0 |
| `t-summary` | Summary | **Tier 1 (Local Gemma)** | Success | 0 | 0 |
| `t-ner` | NER | **Tier 1 (Local Gemma)** | Success | 0 | 0 |
| `t-codedebug` | Code Debug | **Tier 3 (kimi-k2p7-code)** | Success | 128 | 80 |
| `t-codegen` | Code Gen | **Tier 3 (kimi-k2p7-code)** | Success | 112 | 245 |
| `t-logic` | Logic | **Tier 3 (minimax-m3)** | Success | 348 | 95 |

### Total Token Footprint:
* **Prompt Tokens**: 948
* **Completion Tokens**: 529
* **Total Token Score**: **1477** (down from 2417 tokens in Category-based routing!)
