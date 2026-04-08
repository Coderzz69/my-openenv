---
title: AgriEnv OpenEnv Server
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - sustainability
  - agriculture
---

# AgriEnv

AgriEnv is a precision agriculture reinforcement learning environment built in the OpenEnv client/server pattern. It simulates greenhouse control under water, nutrient, pest, and energy tradeoffs, and exposes the standard OpenEnv `reset()`, `step()`, and `state()` interface over HTTP and WebSocket.

## Why this version is the final product

This repo now follows the OpenEnv course architecture instead of only shipping a local Gym-style simulator:

- `agri_env/models.py`: typed OpenEnv contracts for `Action`, `Observation`, and `AgriState`
- `agri_env/client.py`: typed `EnvClient` for training code and notebooks
- `agri_env/env.py`: local deterministic simulator used by the server and baseline agent
- `server/agri_environment.py`: OpenEnv server wrapper
- `server/app.py`: FastAPI app exposing `/reset`, `/step`, `/state`, `/ws`, `/docs`, `/health`, `/metadata`
- `Dockerfile`: Space-ready container entrypoint
- `openenv.yaml`: OpenEnv manifest
- `pyproject.toml`: package metadata and `server` entry point

## Problem motivation

Greenhouse operators need to manage irrigation, NPK dosing, CO2 enrichment, and pesticide use while staying cost-efficient and stable under noisy sensing and weather drift. AgriEnv turns that real control problem into a deterministic, typed RL environment that can be trained locally or deployed as an OpenEnv microservice.

## Observation, action, and reward design

### Observation

`Observation` includes:

- `soil_moisture`
- `nitrogen`
- `phosphorus`
- `potassium`
- `temperature_c`
- `humidity`
- `pest_density`
- `energy_price`
- `water_budget_remaining`
- `growth_stage_progress`

It also includes task metadata and uses the inherited OpenEnv `done`, `reward`, and `metadata` fields.

### Action

`Action` controls:

- `irrigation`
- `nitrogen_injection`
- `phosphorus_injection`
- `potassium_injection`
- `co2_ppm`
- `pesticide`

### Reward

The shaped reward balances:

- crop growth
- moisture alignment
- nutrient alignment
- efficiency bonus
- stability bonus
- task-specific bonus
- operational cost
- overuse penalties
- pest pressure

The scalar reward is returned through the OpenEnv observation contract, and a detailed reward breakdown is stored in observation metadata.

## Tasks

- `easy`: maintain soil moisture near `0.70`
- `medium`: balance NPK and reduce pest pressure
- `hard`: full yield-cost optimization under noise, drift, and water-budget pressure

Each task has a deterministic grader in `[0, 1]`.

The grader score combines:

- cumulative yield
- efficiency (`yield / cost`)
- stability
- task-specific control quality such as moisture accuracy, nutrient balance, pest suppression, and budget retention

## OpenEnv usage

### Local server

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m server.app --port 8000
```

### Validate the environment

```bash
.venv/bin/openenv validate .
.venv/bin/openenv validate --url http://127.0.0.1:8000
python3 -m unittest discover -s tests -v
```

### Use the typed client

```python
from agri_env import Action
from agri_env.client import AgriEnvClient

with AgriEnvClient(base_url="http://127.0.0.1:8000").sync() as env:
    result = env.reset(task="easy", seed=11)
    result = env.step(
        Action(
            irrigation=120.0,
            nitrogen_injection=0.05,
            phosphorus_injection=0.04,
            potassium_injection=0.05,
            co2_ppm=500.0,
            pesticide=0.02,
        )
    )
    print(result.observation.soil_moisture, result.reward, result.done)
```

### Run the baseline locally

```bash
python3 inference.py --policy baseline
```

By default, `inference.py` runs the deterministic baseline across all three tasks and emits one `[START] ... [STEP] ... [END]` block per task.

### Run the baseline against a running OpenEnv server

```bash
export HF_TOKEN=dummy
python3 inference.py --task hard --policy baseline --base-url http://127.0.0.1:8000
```

### Run the LLM controller

```bash
export HF_TOKEN=your_real_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python3 inference.py --task hard --policy llm --base-url http://127.0.0.1:8000
```

The inference runner reads:

- `API_BASE_URL` with a default of `https://router.huggingface.co/v1`
- `MODEL_NAME` with a default of `meta-llama/Llama-3.1-8B-Instruct`
- `HF_TOKEN` with no default

For compatibility with common local setups, it also accepts `OPENAI_API_KEY` or `API_KEY` as fallbacks for the token.

## Deployment

### Docker

```bash
docker build -t agri-env .
docker run --rm -p 8000:8000 agri-env
```

### Hugging Face Spaces

```bash
.venv/bin/openenv push --repo-id <your-username>/agri-env
```

After deployment:

- app UI: `https://<your-username>-agri-env.hf.space/web`
- docs: `https://<your-username>-agri-env.hf.space/docs`
- health: `https://<your-username>-agri-env.hf.space/health`
- websocket endpoint: `wss://<your-username>-agri-env.hf.space/ws`

## Example baseline results

Default seeded baseline scores:

- `easy`: `0.8470`
- `medium`: `0.8084`
- `hard`: `0.8425`

Example final log line format:

```text
[END] success=true steps=140 score=0.843 rewards=0.64,0.68,0.70,...
```

## Notes

- The environment is deterministic for a fixed task and seed.
- The server supports concurrent sessions.
- The Space container enables the OpenEnv web interface through `ENABLE_WEB_INTERFACE=true`.
