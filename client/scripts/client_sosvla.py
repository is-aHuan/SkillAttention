import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json, asyncio, websockets
import cv2
import numpy as np
from typing import List, Any

import pyrealsense2 as rs
from arm.arm import my_Arm

# Updated image size
IMG_HEIGHT = 240
IMG_WIDTH = 320
CAM_HEAD_SERIAL = "335222074941"
CAM_WRIST_SERIAL = "243122076407"
ROBOT_INIT_JOINTS = [-35.407, 5.368, -116.107, -4.17, -34.731, 5.578]

# Remote inference server URI
SERVER_URI = "ws://127.0.0.1:8080/ws/infer"

def euler_xyz_to_matrix(euler):
    """
    euler: [..., 3], XYZ order, units: radians
    Returns: [..., 3, 3] rotation matrix
    """
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # R = Rz * Ry * Rx 
    R = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float32)

    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr

    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr

    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr

    return R

def matrix_to_rot6d(R):
    """
    R: [..., 3, 3]
    rot6d: [..., 6], take the first two columns and flatten
    """
    rot6d = R[..., :, :2]   # [..., 3, 2]
    rot6d = rot6d.reshape(R.shape[:-2] + (6,))
    return rot6d.astype(np.float32)

def euler_to_rot6d(euler):
    R = euler_xyz_to_matrix(euler)
    return matrix_to_rot6d(R)

def euler_xyz_to_quat(euler):
    """
    euler: [..., 3], assumed XYZ order, units: radians (supports list/NumPy input)
    Returns: [..., 4], quaternion (w, x, y, z)
    """
    # Key fix: convert input to a NumPy array first
    euler = np.array(euler, dtype=np.float32)
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat = np.stack([qw, qx, qy, qz], axis=-1)
    return quat.astype(np.float32)

# Data acquisition
def _convert_frame(frame: rs.video_frame, height: int, width: int) -> np.ndarray:
    arr = np.asanyarray(frame.get_data())  # (H, W, 3) RGB
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    arr = arr.transpose(2, 0, 1).astype(np.uint8)
    return arr

def get_head_state(head_frame: rs.video_frame, height: int, width: int) -> List[List[List[float]]]:
    head = _convert_frame(head_frame, height, width)
    head_np = (head.astype(np.float32) / 255.0)
    head_state = [[[float(val) for val in row] for row in channel] for channel in head_np.tolist()]
    return head_state

def get_wrist_state(wrist_frame: rs.video_frame, height: int, width: int) -> List[List[List[float]]]:
    wrist = _convert_frame(wrist_frame, height, width)
    wrist_np = (wrist.astype(np.float32) / 255.0)
    wrist_state = [[[float(val) for val in row] for row in channel] for channel in wrist_np.tolist()]
    return wrist_state

def get_robot_proprio(robot: my_Arm) -> List[float]:
    _, pose = robot.getState()
    quat = euler_xyz_to_quat(pose[3:6])
    grip_pos = robot.get_gripper_position()
    proprio = pose[:3] + [float(q) for q in quat.tolist()] + [float(grip_pos)]
    return proprio

# Initialize cameras
def init_cameras():
    head_pipe = rs.pipeline()
    head_cfg = rs.config()
    head_cfg.enable_device(CAM_HEAD_SERIAL)
    head_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    head_pipe.start(head_cfg)

    wrist_pipe = rs.pipeline()
    wrist_cfg = rs.config()
    wrist_cfg.enable_device(CAM_WRIST_SERIAL)
    wrist_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    wrist_pipe.start(wrist_cfg)
    return head_pipe, wrist_pipe

# Initialize robot
def init_robot():
    robot = my_Arm(wifi=False)
    robot.setJoints(ROBOT_INIT_JOINTS, v=30, r=0, connect=0, block=1)
    robot.setGripperRelease()
    return robot

# Send actions to robot
def execute_action_chunk(robot: my_Arm, actions: List[Any]):
    for idx, act in enumerate(actions):
        arr = np.array(act, dtype=np.float32)
        if arr.shape[0] < 7:
            continue
        joint_rads = arr[:6]
        gripper_val = float(arr[6])

        joint_degs = (joint_rads * 180.0 / np.pi).tolist()
        robot.setPose_CANFD(joint_degs, v=10, r=0, connect=0, block=1)

        grip_pos = int(np.clip(gripper_val, 0, 1) * 1000)
        robot.setGripperPosition(grip_pos)


        print(f"[Action {idx}] joints(deg)={joint_degs} gripper={grip_pos}")

# Remote inference loop
async def loop():
    # Initialize cameras and robot
    head_pipe, wrist_pipe = init_cameras()
    robot = init_robot()

    language_task = "pick up the blue cube"  # language command
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH

    first_request = True
    actions = []

    # Initialize websocket connection
    async with websockets.connect(SERVER_URI, ping_interval=30, ping_timeout=60, max_size=20 * 1024 * 1024) as ws:
        while True:
            if first_request:
                first_request = False

                head_frames = head_pipe.wait_for_frames()
                wrist_frames = wrist_pipe.wait_for_frames()
                head_color = head_frames.get_color_frame()
                wrist_color = wrist_frames.get_color_frame()

                # Get image data and convert to format suitable for sending
                head_state = get_head_state(head_color, img_height, img_width)
                wrist_state = get_wrist_state(wrist_color, img_height, img_width)
                proprio = get_robot_proprio(robot)

                # Send data to inference server
                await ws.send(json.dumps({"head": head_state, 
                                          "wrist": wrist_state, 
                                          "proprio": proprio, 
                                          "lang": language_task, 
                                          "request_action": False})
                                          )
                
                # Receive inference result
                resp = json.loads(await ws.recv())
                if not resp.get("ok"):
                    print("Inference error:", resp)
                    await asyncio.sleep(0.5)
                    continue

                # Get the latest camera frames
                head_frames = head_pipe.wait_for_frames()
                wrist_frames = wrist_pipe.wait_for_frames()
                head_color = head_frames.get_color_frame()
                wrist_color = wrist_frames.get_color_frame()

                # Get image data and convert to format suitable for sending
                head_state = get_head_state(head_color, img_height, img_width)
                wrist_state = get_wrist_state(wrist_color, img_height, img_width)
                proprio = get_robot_proprio(robot)

                # Send data to inference server
                await ws.send(json.dumps({"head": head_state, 
                                        "wrist": wrist_state, 
                                        "proprio": proprio, 
                                        "lang": language_task, 
                                        "request_action": True})
                                        )

                # Receive inference result
                resp = json.loads(await ws.recv())
                if not resp.get("ok"):
                    print("Inference error:", resp)
                    await asyncio.sleep(0.5)
                    continue

            action_recieve = resp.get("action", [])
            if not isinstance(action_recieve, list):
                actions = [action_recieve]
            else:
                actions = action_recieve

            # Execute actions
            for idx, act in enumerate(actions):
                arr = np.array(act, dtype=np.float32)
                if arr.shape[0] < 7:
                    continue
                pose = arr[:6]
                gripper_val = float(arr[6])

                robot.setPose_CANFD(pose)

                grip_pos = int(np.clip(gripper_val, 0, 1) * 1000)
                robot.setGripperPosition(grip_pos)

                print(f"[Action {idx}] pose={pose} gripper={grip_pos}")

                if idx == len(actions) - 2:
                    head_frames = head_pipe.wait_for_frames()
                    wrist_frames = wrist_pipe.wait_for_frames()
                    head_color = head_frames.get_color_frame()
                    wrist_color = wrist_frames.get_color_frame()

                    # Get image data and convert to format suitable for sending
                    head_state = get_head_state(head_color, img_height, img_width)
                    wrist_state = get_wrist_state(wrist_color, img_height, img_width)
                    proprio = get_robot_proprio(robot)

                    # Send data to inference server
                    await ws.send(json.dumps({"head": head_state, 
                                            "wrist": wrist_state, 
                                            "proprio": proprio, 
                                            "lang": language_task, 
                                            "request_action": False})
                                            )
                    # Receive inference result
                    resp = json.loads(await ws.recv())
                    if not resp.get("ok"):
                        print("Inference error:", resp)
                        await asyncio.sleep(0.5)
                        continue

            actions = []
            # Update camera states and continue
            head_frames = head_pipe.wait_for_frames()
            wrist_frames = wrist_pipe.wait_for_frames()
            head_color = head_frames.get_color_frame()
            wrist_color = wrist_frames.get_color_frame()
            next_head_state = get_head_state(head_color, img_height, img_width)
            next_wrist_state = get_wrist_state(wrist_color, img_height, img_width)
            proprio = get_robot_proprio(robot)

            # Continue requesting the next actions
            await ws.send(json.dumps({
                "head": next_head_state,
                "wrist": next_wrist_state,
                "proprio": proprio,
                "lang": language_task,
                "request_action": True
            }))

            resp = json.loads(await ws.recv())
            if not resp.get("ok"):
                print("Inference error:", resp)
                await asyncio.sleep(0.5)
                continue

            action_recieve = resp.get("action", [])
            if not isinstance(action_recieve, list):
                actions = [action_recieve]
            else:
                actions = action_recieve

if __name__ == "__main__":
    asyncio.run(loop())
