# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""平衡机器人强化学习环境实现"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Imu
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
import sys
import os
import numpy as np

# 添加user/test_code到路径以便导入VMC
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../..'))
from user.test_code.test_robot_jointsNsensors import quat_to_euler
from user.test_code.VMC import VMCSolver

from .balance_robot_env_cfg import BalanceRobotEnvCfg


class BalanceRobotEnv(DirectRLEnv):
    """平衡机器人强化学习环境
    
    任务目标：控制平衡机器人保持直立并可能移动
    """
    cfg: BalanceRobotEnvCfg

    def __init__(self, cfg: BalanceRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # TODO: 根据你的实际关节名称获取关节索引
        # 方法1: 如果你知道确切的关节名称
        self._left_front_idx, _ = self.robot.find_joints("Left_front_joint")
        self._left_rear_idx, _ = self.robot.find_joints("Left_rear_joint")
        self._right_front_idx, _ = self.robot.find_joints("Right_front_joint")
        self._right_rear_idx, _ = self.robot.find_joints("Right_rear_joint")
        self._left_wheel_idx, _ = self.robot.find_joints("Left_Wheel_joint")
        self._right_wheel_idx, _ = self.robot.find_joints("Right_Wheel_joint")
        self._controlled_joint_indices = [
            self._left_front_idx, self._left_rear_idx,
            self._right_front_idx, self._right_rear_idx,
            self._left_wheel_idx, self._right_wheel_idx]

        # ['Left_front_joint', 'Left_rear_joint', 'Right_front_joint', 
        #                       'Right_rear_joint', 'Left_Wheel_joint', 'Right_Wheel_joint']
        
        # 方法2: 获取所有关节（如果所有关节都要控制）
        # self._controlled_joint_indices = list(range(self.robot.num_joints))
        
        # 临时：获取所有关节索引（你需要根据实际情况修改）
        # self._controlled_joint_indices = list(range(self.cfg.action_space))
        
        # 缓存关节数据引用
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # 用于计算动作平滑奖励
        self.previous_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), device=self.device
        )
        
        # 初始化VMC求解器 - 每个环境一个左右VMC
        # 参数需要根据实际机器人调整
        self.left_vmc_solvers = [VMCSolver() for _ in range(self.num_envs)]
        self.right_vmc_solvers = [VMCSolver() for _ in range(self.num_envs)]
        
        # 初始化目标速度（每个环境一个）
        self.target_velocity = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """设置场景"""
        # 创建机器人
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # 创建IMU传感器
        self.imu = Imu(self.cfg.imu_cfg)
        
        # 添加地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # 克隆和复制环境
        self.scene.clone_environments(copy_from_source=False)
        
        # CPU仿真需要显式过滤碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # 将机器人添加到场景
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["imu"] = self.imu
        
        # 添加光照
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """在物理步进前处理动作"""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """应用动作到机器人"""
        # 将神经网络输出的动作（通常在[-1, 1]范围）缩放到实际扭矩
        scaled_actions = self.actions * self.cfg.action_scale
        
        # TODO: 根据你的控制方案修改
        # 方案1: 控制所有关节
        self.robot.set_joint_effort_target(scaled_actions)
        
        # 方案2: 只控制特定关节
        # self.robot.set_joint_effort_target(
        #     scaled_actions, 
        #     joint_ids=self._controlled_joint_indices
        # )

    def _get_observations(self) -> dict:
        """获取观察"""
        
        # TODO: 根据你的需求构建观察向量
        # 以下是一个示例，包含IMU数据和关节状态
        
        obs_list = []

        # 观察空间包含：
        # - IMU数据: RPY(3) + 角速度(3) = 6
        # - VMC数据: 双腿摆角平均值(1) + 双腿长度平均值(1) = 2
        # 总计: 6 + 2 = 8维
        
        # 1. IMU数据（6维）
        # RPY (3)
        quat_copy = self.imu.data.quat_w
        rpy = quat_to_euler(quat_copy)
        obs_list.append(rpy)
        # 角速度 (3)
        ang_vel = self.imu.data.ang_vel_b
        obs_list.append(ang_vel)
        
        # 保存RPY和角速度供reward函数使用
        self.roll = rpy[:, 0]      # [num_envs]
        self.pitch = rpy[:, 1]     # [num_envs]
        self.yaw = rpy[:, 2]       # [num_envs]
        self.ang_vel_x = ang_vel[:, 0]  # [num_envs]
        self.ang_vel_y = ang_vel[:, 1]  # [num_envs]
        self.ang_vel_z = ang_vel[:, 2]  # [num_envs]
        
        # 2. VMC数据：计算双腿摆角和长度的平均值 (2维)
        avg_pendulum_angle = torch.zeros((self.num_envs, 1), device=self.device)
        avg_pendulum_length = torch.zeros((self.num_envs, 1), device=self.device)
        avg_pendulum_angle_vel = torch.zeros((self.num_envs, 1), device=self.device)
        avg_pendulum_length_vel = torch.zeros((self.num_envs, 1), device=self.device)
        
        for env_id in range(self.num_envs):
            # 获取关节位置
            left_front_pos = self.joint_pos[env_id, self._left_front_idx].item()
            right_front_pos = self.joint_pos[env_id, self._right_front_idx].item()
            left_rear_pos = self.joint_pos[env_id, self._left_rear_idx].item()
            right_rear_pos = self.joint_pos[env_id, self._right_rear_idx].item()

            left_4_vel = self.joint_vel[env_id, self._left_front_idx].item()
            right_4_vel = self.joint_vel[env_id, self._right_front_idx].item()
            left_1_vel = self.joint_vel[env_id, self._left_rear_idx].item()
            right_1_vel = self.joint_vel[env_id, self._right_rear_idx].item()
            
            # 使用VMC求解器计算,反过来
            self.left_vmc_solvers[env_id].Resolve(math.pi + right_rear_pos, -right_front_pos)
            self.right_vmc_solvers[env_id].Resolve(math.pi - left_rear_pos, left_front_pos)
            left_leg_vel, left_theta_vel = self.left_vmc_solvers[env_id].VMCVelCal(np.array([right_1_vel, -right_4_vel]))
            right_leg_vel, right_theta_vel = self.right_vmc_solvers[env_id].VMCVelCal(np.array([-left_1_vel, left_4_vel]))
            
            # 获取倒立摆参数
            left_angle = self.left_vmc_solvers[env_id].GetPendulumRadian()
            right_angle = self.right_vmc_solvers[env_id].GetPendulumRadian()
            left_length = self.left_vmc_solvers[env_id].GetPendulumLen()
            right_length = self.right_vmc_solvers[env_id].GetPendulumLen()            
            
            # 计算平均值
            avg_pendulum_angle[env_id, 0] = (left_angle + right_angle) / 2.0
            avg_pendulum_length[env_id, 0] = (left_length + right_length) / 2.0
            avg_pendulum_angle_vel[env_id, 0] = (left_theta_vel + right_theta_vel) / 2.0
            avg_pendulum_length_vel[env_id, 0] = (left_leg_vel + right_leg_vel) / 2.0
        
        obs_list.append(avg_pendulum_angle)
        obs_list.append(avg_pendulum_length)
        obs_list.append(avg_pendulum_angle_vel)
        obs_list.append(avg_pendulum_length_vel)

        
        # 保存VMC参数供奖励函数使用
        self.avg_pendulum_angle = avg_pendulum_angle.squeeze(-1)  # [num_envs]
        self.avg_pendulum_length = avg_pendulum_length.squeeze(-1)  # [num_envs]
        self.avg_pendulum_angle_vel = avg_pendulum_angle_vel.squeeze(-1)  # [num_envs]
        self.avg_pendulum_length_vel = avg_pendulum_length_vel.squeeze(-1)  # [num_envs]
        
        # 3. X and V
        obs_list.append(self.robot.data.root_pos_w[:, 0:1])  # X position
        obs_list.append(self.robot.data.root_lin_vel_w[:, 0:1])  # X velocity
        
        # 保存X速度供奖励函数使用
        self.x_velocity = self.robot.data.root_lin_vel_w[:, 0]  # [num_envs]
        
        # 4. 目标速度
        obs_list.append(self.target_velocity.unsqueeze(-1))  # [num_envs, 1]
        
        # 拼接所有观察
        obs = torch.cat(obs_list, dim=-1)
        
        # 确保观察维度正确
        assert obs.shape[1] == self.cfg.observation_space, (
            f"观察维度不匹配！期望 {self.cfg.observation_space}，实际 {obs.shape[1]}"
        )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算奖励"""
        
        # TODO: 根据你的任务目标设计奖励函数
        # 以下是一些常见的奖励项示例
        
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 1. 存活奖励：每步都给予，鼓励机器人保持不倒
        total_reward += self.cfg.rew_scale_alive
        
        # 2. 腿摆角奖励：鼓励保持目标摆角
        target_leg_angle = 1.57  # 目标摆角（约90度）
        
        # theta and theta_dot
        leg_angle_error = torch.abs(self.avg_pendulum_angle - target_leg_angle)
        leg_angle_vel_error = torch.abs(self.avg_pendulum_angle_vel)
        total_reward -= self.cfg.rew_scale_leg_angle * leg_angle_error
        total_reward -= self.cfg.rew_scale_leg_angle_vel * leg_angle_vel_error
        
        # 3. pitch and pitch_dot
        pitch_diff = torch.abs(self.pitch)
        pitch_vel_diff = torch.abs(self.ang_vel_y)
        total_reward -= self.cfg.rew_scale_upright * pitch_diff
        total_reward -= self.cfg.rew_scale_upright_vel * pitch_vel_diff
        
        # 4. 速度跟随奖励：鼓励跟随目标速度
        velocity_error = torch.abs(self.x_velocity - self.target_velocity)
        total_reward -= self.cfg.rew_scale_velocity_tracking * velocity_error
        
        # 5. 速度惩罚：惩罚过大的线速度和角速度
        lin_vel_penalty = -torch.sum(self.robot.data.root_lin_vel_w ** 2, dim=-1)
        total_reward += self.cfg.rew_scale_lin_vel * lin_vel_penalty
        
        ang_vel_penalty = -torch.sum(self.robot.data.root_ang_vel_w ** 2, dim=-1)
        total_reward += self.cfg.rew_scale_ang_vel * ang_vel_penalty
        
        # 4. 关节速度惩罚：惩罚过大的关节速度
        joint_vel_penalty = -torch.sum(
            self.joint_vel[:, self._controlled_joint_indices] ** 2, dim=-1
        )
        total_reward += self.cfg.rew_scale_joint_vel * joint_vel_penalty
        
        # 5. 动作平滑奖励：惩罚动作的突变
        action_rate_penalty = -torch.sum((self.actions - self.previous_actions) ** 2, dim=-1)
        total_reward += self.cfg.rew_scale_action_rate * action_rate_penalty
        self.previous_actions[:] = self.actions
        
        # 6. 扭矩惩罚：惩罚大扭矩（节能）
        torque_penalty = -torch.sum(self.actions ** 2, dim=-1)
        total_reward += self.cfg.rew_scale_torque * torque_penalty
        
        # 7. 终止惩罚：如果环境终止（跌倒），给予额外惩罚
        total_reward += self.cfg.rew_scale_terminated * self.reset_terminated.float()
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判断是否终止"""
        
        # 更新关节状态
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # 1. 超时终止
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 2. 跌倒终止：pitch或roll角度过大
        quat = self.imu.data.quat_w
        w, x, y, z = quat.unbind(-1)
        
        # 计算pitch和roll
        pitch = self.pitch
        roll = self.roll
        
        # 如果倾斜角度过大，认为跌倒
        tipped_over = (torch.abs(pitch) > self.cfg.max_tilt_angle) | \
                      (torch.abs(roll) > self.cfg.max_tilt_angle)
        
        # 3. 位置越界终止：机器人移动太远
        out_of_bounds = torch.any(
            torch.abs(self.robot.data.root_pos_w[:, :2]) > self.cfg.max_position, 
            dim=1
        )
        
        # 组合所有终止条件
        terminated = tipped_over | out_of_bounds
        
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)

        # TODO: 根据需要自定义重置逻辑
        
        # 1. 重置关节状态（添加随机化）
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        
        # 给关节位置添加小的随机扰动
        joint_pos += sample_uniform(
            self.cfg.initial_joint_pos_range[0],
            self.cfg.initial_joint_pos_range[1],
            joint_pos.shape,
            device=self.device,
        )
        
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 2. 重置根状态（添加姿态随机化）
        root_state = self.robot.data.default_root_state[env_ids].clone()
        
        # 给初始姿态添加小的随机扰动
        # 这里简化处理，实际可能需要更复杂的四元数操作
        # TODO: 如果需要姿态随机化，需要实现四元数的随机旋转
        
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
        
        # 3. 重置动作缓存
        self.previous_actions[env_ids] = 0.0
        
        # 4. 随机化目标速度（从target_velocity_range范围内采样）
        self.target_velocity[env_ids] = sample_uniform(
            self.cfg.target_velocity_range[0],
            self.cfg.target_velocity_range[1],
            (len(env_ids),),
            device=self.device,
        )
