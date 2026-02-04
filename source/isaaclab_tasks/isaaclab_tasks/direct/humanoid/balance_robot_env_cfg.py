# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""平衡机器人强化学习环境配置 - RoboMaster Balance Robot"""

import os
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils


from .cod import COD_CFG

@configclass
class WbrRLEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 6  # 6 joints
    observation_space = 31  # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + commands(4) + dof_pos(6) + dof_vel(6) + actions(6)
    state_space = 0

    # viewer - camera position configuration
    # eye: camera position, lookat: target position camera looks at
    # To be on the right side of origin, looking in -Y direction:
    # - Position camera at positive X (right of origin), slightly above (positive Z)
    # - Look at a point in negative Y direction
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.3, 0.0, 0.5),      # 原点右侧2.3米，高度0.5米
        lookat=(2.3, -5.0, 0.3),  # 沿着-Y方向看（向前看）
        origin_type="world"
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = COD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        10.0,  # Left_front_joint
        10.0,  # Left_rear_joint
        10.0,  # Right_front_joint
        10.0,  # Right_rear_joint
        5.0,   # Left_Wheel_joint
        5.0,   # Right_Wheel_joint
    ]

    # Reward parameters - only for pitch stabilization
    alpha_reward_scale: float = 1.2
    pitch_reward_scale: float = 2.0
    actions_cost_scale: float = 0.01
    dof_vel_scale: float = 0.05
    alpha_dot_scale: float = 0.05
    
    death_cost: float = -1.0
    termination_height: float = 0.1

    #commands random configuration
    vx_cmd_range: tuple = (-2.0, 2.0)  # Forward velocity command range
    vy_cmd_range: tuple = (-1.0, 1.0)  # Lateral velocity command range
    wz_cmd_range: tuple = (-2.0, 2.0)  # Yaw rate command range
    height_cmd_range: tuple = (0.15, 0.4)  # Height command range

    # Reward scales
    rew_scale_lin_x: float = 0.5
    rew_scale_lin_y: float = 0.5
    rew_scale_ang_z: float = 0.5
    rew_scale_height: float = -5.0
    rew_scale_projected_gravity: float = -60.0
    rew_scale_lin_z: float = -2.0
    rew_scale_ang_xy: float = -0.05
    rew_scale_joint_acc: float = -2.5e-7
    rew_scale_wheel_acc: float = -0.01
    rew_scale_dof_acc: float = -2.5e-8
    rew_scale_dof_vel: float = -0.00001
    rew_scale_collision: float = -0.01
    rew_scale_calf_sym: float = 0.5
    rew_scale_feet_dist: float = -1.0
