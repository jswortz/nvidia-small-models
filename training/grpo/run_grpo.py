"""Phase 6, step 4: launch NeMo RL GRPO with the router reward environment.

Registers RouterEnvironment with NeMo RL, then runs NeMo RL's own examples/run_grpo.py
with the same command-line arguments. Run inside the nemo-rl container from the repo root:
    python training/grpo/run_grpo.py --config training/configs/router_grpo.yaml
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"  # Ray workers import training.*

from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY  # noqa: E402
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES  # noqa: E402
from nemo_rl.environments.utils import register_env  # noqa: E402

FQN = "training.grpo.router_env.RouterEnvironment"
register_env("router", FQN)
ACTOR_ENVIRONMENT_REGISTRY[FQN] = PY_EXECUTABLES.SYSTEM  # no extra dependencies

NEMO_RL = Path(os.getenv("NEMO_RL_DIR", "/opt/nemo-rl"))
sys.argv[0] = str(NEMO_RL / "examples" / "run_grpo.py")
runpy.run_path(sys.argv[0], run_name="__main__")
