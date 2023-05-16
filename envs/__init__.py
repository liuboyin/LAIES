from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
