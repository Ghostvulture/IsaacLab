"""
Virtual Model Control (VMC) Solver for Balance Robot
虚拟模型控制求解器
"""

import numpy as np
import math


class VMCSolver:
    """VMC求解器 - 用于计算五连杆机构的倒立摆参数"""
    
    def __init__(self, L1=0.15, L4=0.15):
        """
        初始化VMC求解器
        
        Args:
            L1: 第一连杆长度 (默认0.15m)
            L4: 第四连杆长度 (默认0.15m)
        """
        self.L1 = L1
        self.L4 = L4
        
        # 倒立摆参数
        self.pendulum_length = 0.0
        self.pendulum_radian = 0.0
        
    def Resolve(self, theta1: float, theta4: float):
        """
        正运动学求解：从关节角度计算倒立摆参数
        
        Args:
            theta1: 第一关节角度 (弧度)
            theta4: 第四关节角度 (弧度)
        """
        # 计算末端位置
        x = self.L1 * math.cos(theta1) + self.L4 * math.cos(theta4)
        y = self.L1 * math.sin(theta1) + self.L4 * math.sin(theta4)
        
        # 计算倒立摆长度和角度
        self.pendulum_length = math.sqrt(x**2 + y**2)
        self.pendulum_radian = math.atan2(x, -y)  # 注意：这里使用(x, -y)以符合倒立摆定义
        
    def VMCVelCal(self, joint_velocities: np.ndarray) -> tuple[float, float]:
        """
        速度雅可比计算：从关节速度计算倒立摆速度
        
        Args:
            joint_velocities: 关节速度数组 [theta1_dot, theta4_dot]
            
        Returns:
            (leg_length_velocity, pendulum_angle_velocity)
        """
        # 简化的速度计算（实际应该使用雅可比矩阵）
        # 这里需要根据具体的机构参数计算
        # 占位实现：
        leg_length_velocity = np.mean(joint_velocities) * 0.1  # 简化
        pendulum_angle_velocity = (joint_velocities[1] - joint_velocities[0]) * 0.5  # 简化
        
        return leg_length_velocity, pendulum_angle_velocity
    
    def GetPendulumLen(self) -> float:
        """获取倒立摆长度"""
        return self.pendulum_length
    
    def GetPendulumRadian(self) -> float:
        """获取倒立摆角度（弧度）"""
        return self.pendulum_radian
    
    def GetPendulumDegree(self) -> float:
        """获取倒立摆角度（度）"""
        return math.degrees(self.pendulum_radian)


if __name__ == "__main__":
    # 测试代码
    solver = VMCSolver()
    
    # 测试正运动学
    theta1 = math.pi * 0.75  # 135度
    theta4 = -math.pi * 0.25  # -45度
    
    solver.Resolve(theta1, theta4)
    
    print(f"倒立摆长度: {solver.GetPendulumLen():.4f} m")
    print(f"倒立摆角度: {solver.GetPendulumRadian():.4f} rad ({solver.GetPendulumDegree():.2f}°)")
    
    # 测试速度计算
    joint_vels = np.array([1.0, -1.0])
    leg_vel, theta_vel = solver.VMCVelCal(joint_vels)
    print(f"腿长速度: {leg_vel:.4f} m/s")
    print(f"摆角速度: {theta_vel:.4f} rad/s")
