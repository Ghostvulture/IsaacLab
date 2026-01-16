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
from isaaclab.sensors import ImuCfg
import isaaclab.sim as sim_utils


@configclass
class BalanceRobotEnvCfg(DirectRLEnvCfg):
    """平衡机器人强化学习环境配置 - RoboMaster Balance Robot"""
    
    # ========== 环境基本配置 ==========
    decimation = 2  # 控制频率降采样：物理步数/控制步数
    episode_length_s = 10.0  # 每个episode的时长（秒）
    
    # ========== 动作和观察空间定义 ==========
    # TODO: 根据你的机器人关节数量修改！
    # 运行 user/test_code/test_robot_jointsNsensors.py 查看：
    # - Joint names: 会列出所有关节名称
    # - Number of joints: 关节数量
    joint_names = ['Left_front_joint', 'Left_rear_joint', 
                   'Right_front_joint', 'Right_rear_joint', 
                   'Left_Wheel_joint', 'Right_Wheel_joint']
    
    action_space = 6  # TODO: 修改为你要控制的关节数量（例如：左轮关节 + 右轮关节 = 2）
    
    # 观察空间包含：
    # - IMU数据: RPY(3) + 角速度(3) = 6
    # - VMC数据: 腿摆角(1) + 腿长度(1) + 摆角速度(1) + 腿长速度(1) = 4
    # - 机器人根状态: x(1) + v(1) = 2
    # - 目标速度: target_v(1) = 1
    # 总计: 6 + 4 + 2 + 1 = 13
    observation_space = 13  
    
    state_space = 0  # 特权信息空间（通常设为0，除非需要asymmetric actor-critic）

    # ========== 仿真配置 ==========
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 物理仿真时间步长：120Hz
        render_interval=decimation  # 渲染间隔
    )

    # ========== 机器人配置 ==========
    # 获取USD文件路径（相对于此配置文件）
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _usd_path = os.path.abspath(
        os.path.join(_current_dir, "../../../../../../../user/usd_file/USD/COD-2026RoboMaster-Balance.usd")
    )
    
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",  # 机器人在场景中的路径
        
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
            pos=(0.0, 0.0, 0.5),  # 初始位置：抬高0.5米避免穿地
            # TODO: 如果需要设置初始关节角度，在这里添加：
            # joint_pos={
            #     "left_wheel_joint": 0.0,
            #     "right_wheel_joint": 0.0,
            # },
        ),
        
        # 执行器配置
        actuators={
            # TODO: 根据你的机器人修改执行器配置
            # 方案1：所有关节使用相同参数
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # 匹配所有关节
                effort_limit=80.0,  # TODO: 根据电机规格修改扭矩限制（Nm）
                stiffness=0.0,  # 使用扭矩控制
                damping=0.0,
            ),
            
            # 方案2：不同关节组使用不同参数（如果需要，取消注释并修改）
            # "wheels": ImplicitActuatorCfg(
            #     joint_names_expr=[".*wheel.*"],  # 匹配包含"wheel"的关节
            #     effort_limit=100.0,
            #     stiffness=0.0,
            #     damping=0.0,
            # ),
        },
    )

    # ========== IMU传感器配置 ==========
    # TODO: 确认你的机器人base_link名称！
    # 运行 user/test_code/test_robot_jointsNsensors.py 查看 Body names
    imu_cfg: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",  # TODO: 修改为你的base body名称
        update_period=0.01,  # 100Hz更新
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        debug_vis=False,  # 训练时关闭可视化以提高性能
    )

    # ========== 场景配置 ==========
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # 并行环境数量（先用小数量测试，稳定后再增加）
        env_spacing=4.0,  # 环境间距（米）
        replicate_physics=True  # 复制物理场景以提高性能
    )

    # ========== 自定义参数 ==========
    
    # TODO: 修改为你要控制的关节名称
    # 运行 user/test_code/test_robot_jointsNsensors.py 查看 Joint names
    # 示例：如果是平衡车，可能有 "left_wheel_joint", "right_wheel_joint"
    controlled_joint_names = ['Left_front_joint', 'Left_rear_joint', 'Right_front_joint', 
                              'Right_rear_joint', 'Left_Wheel_joint', 'Right_Wheel_joint']
    
    # 动作缩放因子
    action_scale = 50.0  # TODO: 根据需要调整（将神经网络输出[-1,1]映射到扭矩值）
    
    # ========== 目标速度跟随配置 ==========
    # 目标速度范围（米/秒）
    target_velocity_range = [-1.0, 1.0]  # 每次reset时从这个范围随机采样目标速度
    
    # ========== 奖励函数权重 ==========
    # TODO: 根据任务目标调整这些权重
    # 奖励设计建议：
    # 1. 存活奖励：鼓励机器人保持不倒
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
    # TODO: 根据任务修改终止条件
    
    # 机器人倾倒角度限制（弧度）
    # 如果pitch或roll超过这个角度，认为跌倒
    max_tilt_angle = 0.5  # 约28度
    
    # 位置边界：机器人根位置的范围限制
    max_position = 10.0  # 超出此范围则终止（米）
    
    # ========== 重置/初始化配置 ==========
    # TODO: 根据需要修改初始状态的随机化范围
    
    # 初始姿态角度范围（弧度）
    initial_tilt_range = [-0.1, 0.1]  # 初始pitch/roll的随机范围
    
    # 初始关节位置范围（弧度）
    initial_joint_pos_range = [-0.1, 0.1]  # 关节初始位置的随机范围


# ========== 简化配置（用于快速测试） ==========
@configclass 
class BalanceRobotEnvCfg_PLAY(BalanceRobotEnvCfg):
    """用于play/evaluation的配置"""
    def __post_init__(self):
        # 减少环境数量以便观察
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # 延长episode时间
        self.episode_length_s = 20.0
        # 启用IMU可视化
        self.imu_cfg.debug_vis = True
