# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

from .h1_env import H1Env,H1EnvCfg

from .balance_robot_env import WbrRLEnv
from .balance_robot_env_cfg import WbrRLEnvCfg 

##
# Register Gym environments.
##

gym.register(
    id="Isaac-H1-Direct-v0",
    entry_point="isaaclab_tasks.direct.humanoid:H1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-COD-Bipedal-Flat-v0",
    entry_point=f"{__name__}.balance_robot_env:WbrRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WbrRLEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-COD-Bipedal-Flat-Play-v0",
#     entry_point=f"{__name__}.balance_robot_env:BalanceRobotEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": CODBipedalFlatEnvCfg_PLAY,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )