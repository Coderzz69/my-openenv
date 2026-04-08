"""OpenEnv inference entrypoint for AgriEnv."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - container installs it.
    OpenAI = None  # type: ignore[assignment]

from agri_env import Action, AgriEnv, AgriEnvClient
from agri_env.graders import grade_episode
from agri_env.tasks import TASKS
from agri_env.utils import compact_json


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BENCHMARK = "agri-env"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


class HeuristicPolicy:
    """Deterministic baseline controller."""

    def __init__(self) -> None:
        self._last_task: str | None = None
        self._last_observation: Any | None = None
        self._last_action: Action | None = None

    def _reset_if_needed(self, task_id: str) -> None:
        if task_id != self._last_task:
            self._last_task = task_id
            self._last_observation = None
            self._last_action = None

    def _stage_profile(self, progress: float) -> tuple[float, dict[str, float], float]:
        if progress < 0.20:
            return 0.67, {"nitrogen": 0.58, "phosphorus": 0.62, "potassium": 0.58}, 0.45
        if progress < 0.50:
            return 0.70, {"nitrogen": 0.70, "phosphorus": 0.64, "potassium": 0.66}, 0.75
        if progress < 0.80:
            return 0.73, {"nitrogen": 0.66, "phosphorus": 0.67, "potassium": 0.74}, 0.95
        return 0.70, {"nitrogen": 0.60, "phosphorus": 0.62, "potassium": 0.72}, 0.55

    def _smooth(self, value: float, previous: float | None, weight: float = 0.72) -> float:
        if previous is None:
            return value
        return weight * value + (1.0 - weight) * previous

    def act(self, observation: Any, task_id: str) -> Action:
        self._reset_if_needed(task_id)

        moisture_target, nutrient_targets, stage_co2_factor = self._stage_profile(observation.growth_stage_progress)
        if task_id == "easy":
            moisture_target = 0.70
        elif task_id == "hard":
            moisture_target -= 0.02 * max(0.0, 0.30 - observation.water_budget_remaining)

        previous_obs = self._last_observation
        moisture_delta = 0.0 if previous_obs is None else observation.soil_moisture - previous_obs.soil_moisture
        pest_delta = 0.0 if previous_obs is None else observation.pest_density - previous_obs.pest_density

        climate_demand = max(observation.temperature_c - 24.0, 0.0) * 34.0 + max(55.0 - observation.humidity, 0.0) * 10.0
        humidity_brake = max(observation.humidity - 72.0, 0.0) * 12.0
        budget_guard = 0.70 + 0.30 * observation.water_budget_remaining
        irrigation_raw = (
            (moisture_target - observation.soil_moisture) * 2600.0
            - moisture_delta * 900.0
            + climate_demand
            - humidity_brake
        )
        if task_id == "easy":
            irrigation_raw *= 0.90
        elif task_id == "hard":
            irrigation_raw *= 0.92
        irrigation = max(0.0, min(2200.0, irrigation_raw)) * budget_guard

        leaching_brake = max(observation.soil_moisture - 0.80, 0.0) * 0.35
        nutrient_scale = 0.74 if task_id != "easy" else 0.50
        nitrogen_injection = max(0.0, min(0.24, (nutrient_targets["nitrogen"] - observation.nitrogen) * nutrient_scale - leaching_brake))
        phosphorus_injection = max(0.0, min(0.22, (nutrient_targets["phosphorus"] - observation.phosphorus) * nutrient_scale - leaching_brake * 0.8))
        potassium_injection = max(0.0, min(0.24, (nutrient_targets["potassium"] - observation.potassium) * nutrient_scale - leaching_brake * 0.9))

        pest_pressure = max(observation.pest_density - 0.10, 0.0) * 1.25
        humidity_pressure = max(observation.humidity - 74.0, 0.0) * 0.008
        pesticide = pest_pressure + humidity_pressure + max(pest_delta, 0.0) * 0.40
        pesticide = max(0.0, min(0.42 if task_id == "hard" else 0.34, pesticide))
        if task_id == "easy":
            pesticide *= 0.35

        low_price_bias = max(0.0, min(1.0, 1.0 - observation.energy_price / 13.0))
        pest_discount = max(0.70, 1.0 - observation.pest_density * 0.35)
        budget_discount = 0.80 + 0.20 * observation.water_budget_remaining
        co2_ppm = 420.0 + 360.0 * low_price_bias * stage_co2_factor * pest_discount * budget_discount
        if task_id == "easy":
            co2_ppm = min(co2_ppm, 650.0)

        previous_action = self._last_action
        action = Action(
            irrigation=self._smooth(irrigation, None if previous_action is None else previous_action.irrigation, weight=0.68),
            nitrogen_injection=self._smooth(
                nitrogen_injection,
                None if previous_action is None else previous_action.nitrogen_injection,
                weight=0.78,
            ),
            phosphorus_injection=self._smooth(
                phosphorus_injection,
                None if previous_action is None else previous_action.phosphorus_injection,
                weight=0.78,
            ),
            potassium_injection=self._smooth(
                potassium_injection,
                None if previous_action is None else previous_action.potassium_injection,
                weight=0.78,
            ),
            co2_ppm=self._smooth(co2_ppm, None if previous_action is None else previous_action.co2_ppm, weight=0.64),
            pesticide=self._smooth(pesticide, None if previous_action is None else previous_action.pesticide, weight=0.74),
        ).clipped()

        self._last_observation = observation
        self._last_action = action
        return action


class OpenAIController:
    """LLM-backed controller with heuristic fallback."""

    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.fallback = HeuristicPolicy()

    def act(self, observation: Any, task_id: str) -> tuple[Action, str | None]:
        if self.client is None:
            return self.fallback.act(observation, task_id), "openai_client_unavailable"

        task = TASKS[task_id]
        prompt = {
            "task": task.title,
            "description": task.description,
            "observation": observation.to_dict(),
            "constraints": {
                "irrigation": [0.0, 3000.0],
                "nitrogen_injection": [0.0, 0.5],
                "phosphorus_injection": [0.0, 0.5],
                "potassium_injection": [0.0, 0.5],
                "co2_ppm": [300.0, 1200.0],
                "pesticide": [0.0, 1.0],
            },
            "response_schema": {
                "irrigation": "float",
                "nitrogen_injection": "float",
                "phosphorus_injection": "float",
                "potassium_injection": "float",
                "co2_ppm": "float",
                "pesticide": "float",
            },
        }
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You control a precision agriculture greenhouse. "
                            "Return only a JSON object with valid continuous actions."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt)},
                ],
            )
            content = completion.choices[0].message.content or "{}"
            payload = json.loads(_extract_json_object(content))
            return Action.from_any(payload).clipped(), None
        except Exception as exc:  # pragma: no cover - networked path.
            return self.fallback.act(observation, task_id), str(exc)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return text[start : end + 1]


def _build_client(api_base_url: str, hf_token: str) -> Any:
    if OpenAI is None or not hf_token:
        return None
    return OpenAI(base_url=api_base_url, api_key=hf_token)


def _print_start(task: str, model_name: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model_name}")


def _format_error_text(message: str | None) -> str:
    if not message:
        return "null"
    line = str(message).replace("\r", " ").replace("\n", " ").strip()
    return line or "null"


def _action_log_payload(action: Action) -> str:
    payload = action.to_dict()
    payload.pop("metadata", None)
    return compact_json(payload)


def _print_step(step: int, action: Action, reward_value: float, done: bool, error: str | None) -> None:
    print(
        "[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}".format(
            step=step,
            action=_action_log_payload(action),
            reward=reward_value,
            done=str(done).lower(),
            error=_format_error_text(error),
        )
    )


def _print_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    formatted_rewards = ",".join(f"{value:.2f}" for value in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={formatted_rewards}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgriEnv inference.")
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--policy", choices=["baseline", "llm"], default="baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-url", default=None, help="Optional OpenEnv server base URL.")
    return parser.parse_args()


def _zero_action() -> Action:
    return Action(
        irrigation=0.0,
        nitrogen_injection=0.0,
        phosphorus_injection=0.0,
        potassium_injection=0.0,
        co2_ppm=400.0,
        pesticide=0.0,
    )


def _grader_score(grader_payload: Any) -> float:
    if isinstance(grader_payload, dict) and "score" in grader_payload:
        try:
            return max(0.0, min(1.0, float(grader_payload["score"])))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _grader_passed(grader_payload: Any, task_id: str) -> bool:
    if isinstance(grader_payload, dict) and "passed" in grader_payload:
        return bool(grader_payload["passed"])
    return _grader_score(grader_payload) >= TASKS[task_id].pass_threshold


def _run_local_task(task_id: str, policy: HeuristicPolicy | OpenAIController, seed: int | None) -> tuple[bool, int, float, list[float]]:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = AgriEnv(task=task_id, seed=seed)
    _print_start(task_id, MODEL_NAME)

    try:
        observation = env.reset(seed=seed, task=task_id)
        for step in range(1, env.task.horizon + 1):
            error_message = None
            if isinstance(policy, HeuristicPolicy):
                action = policy.act(observation, task_id)
            else:
                action, error_message = policy.act(observation, task_id)
            observation, reward, done, info = env.step(action)
            reward_value = float(reward.total)
            rewards.append(reward_value)
            steps_taken = step
            _print_step(step, action, reward_value, done, error_message)
            if done:
                grader_payload = info.get("grader", {})
                score = _grader_score(grader_payload)
                success = _grader_passed(grader_payload, task_id)
                break

        if steps_taken and score == 0.0:
            grader_payload = grade_episode(env.episode_summary()).to_dict()
            score = _grader_score(grader_payload)
            success = _grader_passed(grader_payload, task_id)
    except Exception as exc:
        success = False
        step = steps_taken + 1 if steps_taken < TASKS[task_id].horizon else TASKS[task_id].horizon
        _print_step(step, _zero_action(), 0.0, True, str(exc))
    finally:
        _print_end(success, steps_taken, score, rewards)

    return success, steps_taken, score, rewards


def _run_remote_task(
    task_id: str,
    policy: HeuristicPolicy | OpenAIController,
    seed: int | None,
    base_url: str,
) -> tuple[bool, int, float, list[float]]:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    _print_start(task_id, MODEL_NAME)

    try:
        with AgriEnvClient(base_url=base_url).sync() as env:
            result = env.reset(task=task_id, seed=seed)
            observation = result.observation
            for step in range(1, TASKS[task_id].horizon + 1):
                error_message = None
                if isinstance(policy, HeuristicPolicy):
                    action = policy.act(observation, task_id)
                else:
                    action, error_message = policy.act(observation, task_id)
                result = env.step(action)
                observation = result.observation
                reward_value = float(result.reward or 0.0)
                rewards.append(reward_value)
                steps_taken = step
                grader_payload = observation.metadata.get("grader", {})
                _print_step(step, action, reward_value, result.done, error_message)
                if result.done:
                    if not grader_payload:
                        grader_payload = env.state().grader
                    score = _grader_score(grader_payload)
                    success = _grader_passed(grader_payload, task_id)
                    break
    except Exception as exc:
        success = False
        step = steps_taken + 1 if steps_taken < TASKS[task_id].horizon else TASKS[task_id].horizon
        _print_step(step, _zero_action(), 0.0, True, str(exc))
    finally:
        _print_end(success, steps_taken, score, rewards)

    return success, steps_taken, score, rewards


def main() -> int:
    _ = LOCAL_IMAGE_NAME
    args = parse_args()
    client = _build_client(api_base_url=API_BASE_URL, hf_token=HF_TOKEN or "")
    policy = HeuristicPolicy() if args.policy == "baseline" else OpenAIController(client=client, model_name=MODEL_NAME)
    task_ids = list(TASKS) if args.task == "all" else [args.task]

    all_success = True
    for task_id in task_ids:
        success, _steps, _score, _rewards = (
            _run_remote_task(task_id, policy, args.seed, args.base_url)
            if args.base_url
            else _run_local_task(task_id, policy, args.seed)
        )
        all_success = all_success and success

    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
