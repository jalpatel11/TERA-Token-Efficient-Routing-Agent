# TERA: Token Efficient Routing Agent

TERA is a Hybrid Token-Efficient Routing Agent built for the AMD ACT II Hackathon (Track 1). It dynamically routes prompts to the most cost-effective and capable Fireworks AI model at runtime to minimize token consumption while maintaining output accuracy.

---

## How It Works

The execution pipeline runs in 6 distinct phases:

1. **Centralized Configuration (`config.py`)**
   Loads and validates environment variables (`FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`) at startup. Enforces strict presence checks.

2. **Model Capability Ranking (`agent/model_selector.py`)**
   Inspects names in `ALLOWED_MODELS` and ranks them from smallest (cheapest) to largest (most capable) using parameter-size heuristic parsing (e.g. matching `8b`, `2p7`, `31b`, etc.). 
   * **Gemma-Only Restriction**: If Gemma models are present in the list, the selector dynamically filters out other models to ensure compliance.
   * **Code-Specialized Priority**: If a task requires code generation or debugging, the selector checks for and routes to a model containing `code` or `coder` in its name.
   * **General Routing**: Simple tasks (Sentiment, Summarization, NER) route to the cheap general model, while complex tasks (Math, Logic, General Reasoning) route to the capable flagship model.

3. **Inexpensive Keyword-based Routing (`agent/router.py`)**
   Parses user prompts against keyword heuristics to classify them into the 8 required categories. This classifier executes locally, taking 0 milliseconds and costing 0 remote API tokens.

4. **API Execution (`agent/fireworks_client.py`)**
   Wraps HTTP POST requests to the Fireworks completions endpoint. It configures standard connection pools and handles transient errors (HTTP 429, 502, 503, 504) using exponential backoff retry cycles.

5. **Deterministic Formatting (`agent/formatter.py`)**
   Normalizes the model outputs depending on task types (extracting JSON entities for NER, removing markdown code fences, and cleaning math output prefixes).

6. **Log & Export Output (`main.py`)**
   Records token usage metrics and latency locally, structures the final predictions, and writes them to the target output file.

---

## How to Configure and Run the Code

### 1. Environment Settings
Create a `.env` file in the root directory (based on `.env.example`).
```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/glm-5p1,accounts/fireworks/models/gpt-oss-120b
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
To package the agent into a lightweight (~150MB) Docker container:
1. Build the image:
   ```bash
   docker build -t tera-agent .
   ```
2. Run the container, mounting the host input/output folders and passing runtime environment variables:
   ```bash
   docker run --rm \
     -v $(pwd)/input:/input \
     -v $(pwd)/output:/output \
     -e FIREWORKS_API_KEY="your_api_key" \
     -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
     -e ALLOWED_MODELS="accounts/fireworks/models/glm-5p1,accounts/fireworks/models/gpt-oss-120b" \
     tera-agent
   ```

---

## Current Version Benchmark Results

We evaluated the pipeline locally on a mock dataset containing one task for each of the 8 capability categories.

### System Configuration used for Test Run:
* **Models**: `accounts/fireworks/models/glm-5p1` (Cheap) and `accounts/fireworks/models/gpt-oss-120b` (Capable)
* **Total Elapsed Latency**: 32.8s (~4.1s average per task)

### Task Execution Details:

| Task ID | Task Category | Routed Model | Latency | Prompt Tokens | Completion Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `t-general` | General | `gpt-oss-120b` | 0.66s | 116 | 56 |
| `t-math` | Math | `gpt-oss-120b` | 0.52s | 124 | 77 |
| `t-sentiment` | Sentiment | `glm-5p1` | 8.30s | 62 | 258 |
| `t-summary` | Summary | `glm-5p1` | 12.83s | 96 | 441 |
| `t-ner` | NER | `glm-5p1` | 8.99s | 67 | 364 |
| `t-codedebug` | Code Debug | `gpt-oss-120b` | 0.48s | 128 | 80 |
| `t-codegen` | Code Gen | `gpt-oss-120b` | 0.55s | 112 | 158 |
| `t-logic` | Logic | `gpt-oss-120b` | 0.49s | 135 | 143 |

### Total Token Footprint:
* **Prompt Tokens**: 840
* **Completion Tokens**: 1577
* **Total Token Score**: 2417
