import ctypes
import math
import logging
import threading
from typing import List, Optional, Tuple

import arm.rm_robot_interface
import numpy as np

from arm.rm_robot_interface import *

logging.basicConfig(level=logging.INFO)
logger_ = logging.getLogger(__name__)

import json

# copy里面的POSE在我们这里面是Pose

Joints = ctypes.c_float * 6
GRIPPER_SPEED = 0.5

def euler2quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ (roll,pitch,yaw) -> quaternion [qx,qy,qz,qw]"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-8
    return list(q)

class my_Arm():
    def __init__(self, wifi=False):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        # self.arm = RoboticArm()

        if not wifi:
            self.handle = self.arm.rm_create_robot_arm("192.168.1.18", 8080)
        else:
            self.handle = self.arm.rm_create_robot_arm("192.168.33.80", 8080)

        self.joints = []
        # self.pose = Pose()

    def algo_set_joint_min_limit(self, joint_limit = [-178.0, -123.0, -178.0, -178.0, -178.0, -360.0]):
        self.arm.rm_algo_set_joint_min_limit(joint_limit = joint_limit)

    def getState(self):
        res = self.arm.rm_get_current_arm_state()
        self.joints = [_ for _ in res[1]['joint']]
        joints = [res[1]['joint'][i] for i in range(6)]
        pose = [res[1]['pose'][i] for i in range(6)]
        # print((joints, pose))
        return joints, pose
    
    def get_robot_state(self):
        res = self.arm.rm_get_current_arm_state()
        joints = [res[1]['joint'][i] for i in range(6)]
        pose = [res[1]['pose'][i] for i in range(6)]
        quaternion = euler2quat(pose[3], pose[4], pose[5])  
        pose = pose[:3] + quaternion  # 四元数格式转换wxyz->xyzw
        return joints, pose # 返回位置+四元数
    
    def get_joints_angles_and_pose(self):
        res = self.arm.rm_get_current_arm_state()
        joints = [res[1]['joint'][i] * M_PI / 180.0 for i in range(6)]
        pose = [res[1]['pose'][i] for i in range(6)]
        return joints, pose
    
    def get_gripper_position(self):
        res = self.arm.rm_get_gripper_state()
        gripper_pos = res[1]['actpos'] / 1000.0

        if gripper_pos < 0.0:
            gripper_pos = 0.0
        elif gripper_pos > 1.0:
            gripper_pos = 1.0

        return gripper_pos

    def setJoints(self, joints, v = 30, **kwargs):
        self.arm.rm_movej(joints, v = v, **kwargs)

    def setJoints_CANFD(self, joints, follow=False):
        # 透传是不过机械臂的算法直接控制关节的运动接口，用于用户自行规划算法，高跟随是机械臂不对路径做优化 下发点位关节就运动
        # 低跟随是机械臂对下发的路径点进行一些插补，保障运行的顺滑
        ret = self.arm.rm_movej_canfd(joints, follow)  # 不确定是不是高跟随----低跟随
        return ret
    
    def setGripperRelease(self, speed=1000, block=True, timeout=4):
        self.arm.rm_set_gripper_release(speed, block, timeout)  # position:0-1000

    def setPose_P(self, pose, speed=20, curvatureR=0, block=1):  # speed:0~100%
        # use joints space,it can go where it can go. And it need palnner, so it cost a very short time to start move
        ret = self.arm.rm_movej_p(pose, speed, r=curvatureR, block=block)
        return ret

    def setPose(self, pose, speed=10, curvatureR=0, connect=0, block=0):  # speed:0~100%
        # use Descartes space,it goes along straight line, so sometimes it can't go someplace it can go
        ret = self.arm.rm_movel(pose, speed, r=curvatureR, connect=connect, block=block)
        return ret

    def setPose_CANFD(self, pose, follow=False):  # speed:0~100%     不确定是否可以6pose
        tag = self.arm.rm_movep_canfd(pose, follow)
        return tag

    def setGripperPosition(self, position, block=True, timeout=4):
        self.arm.rm_set_gripper_position(position, block, timeout)  # position:0-1000

    def get_gripper_state(self):
        res = self.arm.rm_get_gripper_state()[1]
        gripper_pos = res['actpos'] / 1000.0

        if gripper_pos < 0.0:
            gripper_pos = 0.0
        elif gripper_pos > 1.0:
            gripper_pos = 1.0

        if res['mode'] == 4:
            gripper_speed = -GRIPPER_SPEED
        elif res['mode'] == 5:
            gripper_speed = GRIPPER_SPEED
        else:
            gripper_speed = 0.0

        return gripper_pos, gripper_speed

    def set_arm_stop(self):
        self.arm.rm_set_arm_stop()
    
    def Get_All_State(self):
        tag, state = self.arm.rm_get_arm_all_state()
        return tag, state
    
    def get_joint_degree(self):
        return self.arm.rm_get_joint_degree()[1]

    def __del__(self):
        self.arm.rm_delete_robot_arm()
        logger_.info(f'关闭与机械臂的Socket连接')

if __name__ == "__main__":
    arm = my_Arm(wifi=False)