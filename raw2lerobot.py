#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
本脚本可将原始数据转换为 LeRobotDataset 格式，
支持批量处理/home/tanner/embodiedAI_ws/data内从202到250文件夹的数据，
所有数据保存为统一目录结构，tasks.jsonl内的task留空。

示例用法：
    python openx_rlds.py \
        --raw-dir /path/to/data \
        --local-dir /path/to/local_dir \
        --repo-id your_id \
        --use-videos \
        --push-to-hub
"""

import argparse
import json
import re
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Literal
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LEFT_INIT_POS = [
    -0.05664911866188049,
    -0.26874953508377075,
    0.5613412857055664,
    1.483367681503296,
    -1.1999313831329346,
    -1.3498512506484985,
    0,
]
RIGHT_INIT_POS = [
    -0.05664911866188049,
    -0.26874953508377075,
    0.5613412857055664,
    -1.483367681503296,
    1.1999313831329346,
    1.3498512506484985,
    0,
]

motors = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

def find_match(list, key):
    for item in list:
        if key in item:
            return item
    return None

def process_custom_format(lerobot_dataset: LeRobotDataset, raw_dir: Path):
    with open(raw_dir / "low_dim.json", "r") as f:
        low_dim = json.load(f)

    ep_len = len(low_dim["action/arm/joint_position"])
    qpos = np.array(low_dim["observation/arm/joint_position"])
    qaction = np.array(low_dim["action/arm/joint_position"])
    gripper_pos = np.array(low_dim["observation/eef/joint_position"])
    gripper_action = np.array(low_dim["action/eef/joint_position"])

    if qpos.shape[-1] == 12:
        state = np.concatenate(
            [qpos[:, :6], gripper_pos[:, 0:1], qpos[:, 6:], gripper_pos[:, 1:2]], axis=1
        )
        action = np.concatenate(
            [qaction[:, :6], gripper_action[:, 0:1], qaction[:, 6:], gripper_action[:, 1:2]], axis=1
        )
    elif qpos.shape[-1] == 6:
        state = np.concatenate(
            [np.tile(LEFT_INIT_POS, (ep_len, 1)), qpos, gripper_pos], axis=1
        )
        action = np.concatenate(
            [np.tile(LEFT_INIT_POS, (ep_len, 1)), qaction, gripper_action], axis=1
        )
    else:
        raise ValueError("Unexpected qpos dimension.")

    # 加载相机图像
    cams = {
        "cam_high": find_match(os.listdir(raw_dir), "cam1"),
        "cam_left_wrist": find_match(os.listdir(raw_dir), "cam2"),
        "cam_right_wrist": find_match(os.listdir(raw_dir), "cam3"),
    }

    imgs_per_cam = {}
    for cam_name, cam_folder_name in cams.items():
        cam_folder_path = raw_dir / cam_folder_name if cam_folder_name else None
        imgs = []
        if cam_folder_path and cam_folder_path.exists():
            for i in range(ep_len):
                img_path = cam_folder_path / f"frame_{i:06}.png"
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
            imgs_per_cam[cam_name] = np.array(imgs)
        else:
            imgs_per_cam[cam_name] = np.zeros((ep_len, 480, 640, 3), dtype=np.uint8)

    # 逐帧添加
    for i in range(ep_len):
        frame_dict = {
            "observation.images.cam_high": imgs_per_cam["cam_high"][i],
            "observation.images.cam_left_wrist": imgs_per_cam["cam_left_wrist"][i],
            "observation.images.cam_right_wrist": imgs_per_cam["cam_right_wrist"][i],
            "observation.state": state[i],
            "action": action[i],
            "task": "",
        }
        lerobot_dataset.add_frame(frame_dict)

    lerobot_dataset.save_episode()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="原始数据父目录，例如：/home/tanner/embodiedAI_ws/data。",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="转换后保存的目录，例如：0323_tanner。",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Hugging Face 仓库ID。",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="转换后上传至Hugging Face Hub。",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="将图片转换为视频。",
    )
    parser.add_argument(
        "--image-writer-process",
        type=int,
        default=5,
        help="图片写入进程数。",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="每个进程的写入线程数。",
    )
    args = parser.parse_args()

    # 获取所有202-250的文件夹
    sub_folders = []
    for f in args.raw_dir.iterdir():
        if f.is_dir() and f.name.isdigit():
            num = int(f.name)
            if 202 <= num <= 204:
                sub_folders.append(f)
    sub_folders = sorted(sub_folders, key=lambda x: int(x.name))

    if not sub_folders:
        print("No folders in range 202-204 found.")
        return

    # 初始化数据集配置
    first_folder = sub_folders[0]
    with open(first_folder / "meta.json", "r") as f:
        meta = json.load(f)
    fps = meta.get("fps", 25)

    # 定义特征
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float64",
            "shape": (len(motors),),
            "names": [motors],
        },
    }
    for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if args.use_videos else "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    # 创建数据集目录
    if args.local_dir.exists():
        shutil.rmtree(args.local_dir)
    lerobot_dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="airbot",
        root=args.local_dir,
        fps=fps,
        use_videos=args.use_videos,
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_process,
    )

    # 批量处理所有文件夹
    for folder in sub_folders:
        process_custom_format(lerobot_dataset, folder)

    # 上传到Hub
    if args.push_to_hub:
        tags = ["LeRobot", "batch_processed"]
        lerobot_dataset.push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    main()