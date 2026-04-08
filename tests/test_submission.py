from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from agri_env import Action, AgriEnv, AgriState, Observation, Reward, grade_episode
from inference import HeuristicPolicy
from server.app import app


ROOT = Path(__file__).resolve().parents[1]
STEP_PATTERN = re.compile(
    r"^\[STEP\] step=\d+ action=\{.*\} reward=-?\d+\.\d{2} done=(true|false) error=.*$"
)
START_PATTERN = re.compile(r"^\[START\] task=(easy|medium|hard) env=agri-env model=.+$")
END_PATTERN = re.compile(
    r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*)?$"
)


class AgriEnvSubmissionTests(unittest.TestCase):
    def rollout(self, task: str, seed: int | None = None) -> tuple[dict, dict]:
        env = AgriEnv(task=task, seed=seed)
        policy = HeuristicPolicy()
        observation = env.reset(seed=seed)
        done = False
        while not done:
            observation, reward, done, info = env.step(policy.act(observation, task))
        summary = env.episode_summary().to_dict()
        grader = grade_episode(env.episode_summary()).to_dict()
        return summary, grader

    def test_reset_step_and_state_contract(self) -> None:
        env = AgriEnv(task="medium", seed=123)
        observation = env.reset(seed=123)
        self.assertIsInstance(observation, Observation)

        action = Action(
            irrigation=180.0,
            nitrogen_injection=0.10,
            phosphorus_injection=0.08,
            potassium_injection=0.09,
            co2_ppm=520.0,
            pesticide=0.05,
        )
        next_observation, reward, done, info = env.step(action)

        self.assertIsInstance(next_observation, Observation)
        self.assertIsInstance(reward, Reward)
        self.assertIsInstance(done, bool)
        self.assertIn("reward_breakdown", info)

        latent_state = env.state()
        self.assertIsInstance(latent_state, AgriState)
        self.assertEqual(latent_state.task_id, "medium")
        self.assertTrue(latent_state.growth_stage)
        self.assertGreaterEqual(latent_state.water_budget_remaining, 0.0)

    def test_seeded_rollout_is_deterministic(self) -> None:
        first_summary, first_grader = self.rollout(task="hard", seed=77)
        second_summary, second_grader = self.rollout(task="hard", seed=77)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_grader, second_grader)

    def test_baseline_passes_all_tasks(self) -> None:
        for task in ("easy", "medium", "hard"):
            with self.subTest(task=task):
                summary, grader = self.rollout(task=task)
                self.assertGreaterEqual(grader["score"], 0.0)
                self.assertLessEqual(grader["score"], 1.0)
                self.assertTrue(grader["passed"])
                self.assertGreater(summary["cumulative_yield"], 0.0)
                self.assertGreater(summary["average_efficiency"], 0.0)

    def test_fastapi_app_exposes_openenv_routes(self) -> None:
        route_paths = {route.path for route in app.routes}
        self.assertIn("/health", route_paths)
        self.assertIn("/metadata", route_paths)
        self.assertIn("/reset", route_paths)
        self.assertIn("/step", route_paths)
        self.assertIn("/state", route_paths)
        self.assertIn("/ws", route_paths)

        schema = app.openapi()
        self.assertIn("/health", schema["paths"])
        self.assertIn("/metadata", schema["paths"])
        self.assertIn("/reset", schema["paths"])
        self.assertIn("/step", schema["paths"])

    def test_inference_baseline_output_format(self) -> None:
        env = os.environ.copy()
        completed = subprocess.run(
            [sys.executable, "inference.py", "--task", "easy", "--policy", "baseline"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertRegex(lines[0], START_PATTERN)
        self.assertRegex(lines[-1], END_PATTERN)
        for line in lines[1:-1]:
            self.assertRegex(line, STEP_PATTERN)
        self.assertIn("done=true", lines[-2])

    def test_inference_default_runs_all_tasks(self) -> None:
        completed = subprocess.run(
            [sys.executable, "inference.py", "--policy", "baseline"],
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        start_tasks = [line.split()[1].split("=", 1)[1] for line in lines if line.startswith("[START]")]
        self.assertEqual(sum(1 for line in lines if line.startswith("[START]")), 3)
        self.assertEqual(sum(1 for line in lines if line.startswith("[END]")), 3)
        self.assertEqual(start_tasks, ["easy", "medium", "hard"])


if __name__ == "__main__":
    unittest.main()
