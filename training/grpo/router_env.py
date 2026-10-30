"""Phase 6, step 4: a NeMo RL environment that scores router outputs with training/reward.py.

GRPO samples several routes for each training message. This environment gives each one
the store-state reward: run lookup, policy engine, and action on a fresh store, and
compare with the store the labeled route produces. Ground truth for each message is
the JSON written by training/grpo/build_grpo_data.py: {"id", "router_input", "expected_route", "group"}.

The actor runs with max_concurrency=1 because reward() uses an environment variable for
the store path and is not thread-safe. One reward takes about 13 ms, so a step of
16 prompts x 8 samples costs under 2 seconds of scoring.
"""
from __future__ import annotations

import json

import ray
import torch
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from training import reward as R


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1)  # pragma: no cover
class RouterEnvironment(EnvironmentInterface):
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def step(self, message_log_batch, metadata):
        rewards, harmful, valid = [], [], []
        for conversation, meta in zip(message_log_batch, metadata):
            output = "".join(str(m["content"]) for m in conversation if m["role"] == "assistant")
            case = json.loads(meta["ground_truth"])
            rewards.append(R.reward(output, case))
            harmful.append(R.is_harmful(output, case))
            valid.append(R.valid_json(output))
        for m, h, v in zip(metadata, harmful, valid):
            m["harmful"], m["valid_json"] = h, v
        observations = [{"role": "environment", "content": f"reward {r:.2f}"} for r in rewards]
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(rewards),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(len(rewards)),
            answers=None,
        )

    def global_post_process_and_metrics(self, batch):
        rewards = batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
        # Full reward for a correct route is 1.0 + 0.25 + 0.1 = 1.35.
        return batch, {
            "mean_reward": rewards.mean().item(),
            "exact_route_rate": (rewards >= 1.35 - 1e-6).float().mean().item(),
            "harmful_rate": (rewards < 0).float().mean().item(),
            "invalid_json_rate": (rewards == 0).float().mean().item(),
            "fraction_properly_ended": batch["is_end"].float().mean().item(),
        }

    def shutdown(self) -> None:
        pass
