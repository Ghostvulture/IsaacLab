# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""平衡机器人强化学习环境配置 - RoboMaster Balance Robot"""

import os
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils


@configclass
class LocomotionBipedalEnvCfg(DirectRLEnvCfg):
    """平衡机器人基础环境配置 - 包含核心观察、动作和奖励参数"""
    
    # env
    episode_length_s = 10.0
    decimation = 2
    action_scale = 50.0
    action_space = 6
    observation_space = 13
    state_space = 0
    
    # ========== 仿真配置 ==========
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 物理仿真时间步长：120Hz
        render_interval=decimation,  # 渲染间隔
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ========== 场景配置 ==========
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, 
        env_spacing=5.0,  # 环境间距
        replicate_physics=True,
        clone_in_fabric=True  # 关键：启用fabric克隆，确保环境正确分布
    )

    # robot configuration
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _usd_path = os.path.abspath(
        os.path.join(_current_dir, "../../../../../user/usd_file/USD/COD-2026RoboMaster-Balance.usd")
    )
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=80.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # target velocity range
    target_velocity_range: tuple[float, float] = (-1.0, 1.0)

    # ========== 奖励权重配置 ==========
    # 1. 存活奖励
    rew_scale_alive = 1.0

    # 2. 腿摆角和摆角速度惩罚
    rew_scale_leg_angle = 1.0
    rew_scale_leg_angle_vel = 0.1
    
    # 速度跟随奖励：鼓励机器人跟随目标速度
    rew_scale_velocity_tracking = 1.0  # 惩罚实际速度与目标速度的偏差
    
    # 2. 终止惩罚：机器人跌倒的惩罚
    rew_scale_terminated = -2.0
    
    # 3. 姿态奖励：鼓励机器人保持直立（pitch角接近0）
    rew_scale_upright = 2.0  # 惩罚pitch角偏离
    rew_scale_upright_vel = 0.1  # 惩罚pitch角速度
    
    # 4. 速度控制：惩罚过大的速度
    rew_scale_lin_vel = -0.01  # 惩罚过大的线速度
    rew_scale_ang_vel = -0.01  # 惩罚过大的角速度
    
    # 5. 关节速度：惩罚过大的关节速度
    rew_scale_joint_vel = -0.001
    
    # 6. 动作平滑：惩罚动作的突变
    rew_scale_action_rate = -0.01
    
    # 7. 能量消耗：惩罚大扭矩
    rew_scale_torque = -0.0001
    
    # ========== 终止条件 ==========
    max_tilt_angle: float = 0.5
    max_position: float = 10.0
    
    # ========== 重置配置 ==========
    initial_tilt_range: tuple[float, float] = (-0.1, 0.1)
    initial_joint_pos_range: tuple[float, float] = (-0.1, 0.1)


@configclass
class CODBipedalFlatEnvCfg(LocomotionBipedalEnvCfg):
    """COD平衡机器人平地环境配置"""
    
    def __post_init__(self):
        """Post initialization - 针对平地环境的参数调整"""
        super().__post_init__()
        
        # 平地环境可以用更多环境并行训练
        self.scene.num_envs = 24
        self.scene.env_spacing = 4.0
        
        # 平地上可以训练更久一些
        self.episode_length_s = 15.0
        
        # 平地上摩擦力设置
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0


@configclass
class CODBipedalFlatEnvCfg_PLAY(CODBipedalFlatEnvCfg):
    """COD平衡机器人平地环境 - Play模式配置"""
    
    def __post_init__(self):
        super().__post_init__()
        # Play模式只用一个环境
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
