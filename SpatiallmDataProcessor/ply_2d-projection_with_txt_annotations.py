#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Author       : ZhangJie
# @ Date         : 2025-11-07
# @ Version      : 1.0.0
"""
====================================================================
PLY 2D-Projection with TXT Annotations
====================================================================
介绍：
自动扫描目录，将场景的多房间PLY点云垂直投影到XY平面，并叠加语义标注生成2D可视化图。

输入：
1. PLY文件所在目录路径
2. TXT标注文件所在目录路径
3. 场景ID、样本ID及房间选择参数

输出：
自动命名的PNG文件（dpi=300），含完整标注信息
--------------------------------------------------------------------
命令行用法

模式1 - 处理单个房间：
$ python script.py <output_dir> <ply_dir> <txt_dir> <scene_id> --sample <id> --room <id>

模式2 - 处理所有房间：
$ python script.py <output_dir> <ply_dir> <txt_dir> <scene_id> --sample <id> --all

示例：
# 处理 scene_000000 sample为1 的单个房间00
$ python script.py output/ data/pcd data/txt scene_000000 --sample 1 --room 00

# 处理 scene_000000 sample为1 的所有房间
$ python script.py output/ data/pcd data/txt scene_000000 --sample 1 --all

# 输出文件自动命名: scene_000000_sample_1_all_rooms.png
"""
import argparse
import sys
from pathlib import Path
import re
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from entity import Wall as _OriginalWall, Door as _OriginalDoor, Window as _OriginalWindow, Bbox as _OriginalBbox

@dataclass
class Wall(_OriginalWall):
    room_id: int = 0  # 添加新字段

    def __post_init__(self):
        super().__post_init__()
        self.room_id = int(self.room_id)


@dataclass
class Door(_OriginalDoor):
    room_id: int = 0 # 新增：标识所属房间

    def __post_init__(self):
        super().__post_init__()
        self.room_id = int(self.room_id)

@dataclass
class Window(Door):
    entity_label: str = "window"

@dataclass
class Bbox(_OriginalBbox):
    room_id: int = 0  # 新增：标识所属房间

    def __post_init__(self):
        super().__post_init__()
        self.room_id = int(self.room_id)

# --------------------------
# 2. 核心功能函数
# --------------------------
def read_ply_and_project(ply_path):
    """读取PLY点云并垂直投影到XY平面（z=0）"""
    # 将Path对象转换为字符串
    pcd = o3d.io.read_point_cloud(str(ply_path))  # 关键修改：添加str()转换
    if not pcd.has_points():
        raise ValueError(f"PLY文件中未检测到点云数据: {ply_path}")
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points[:, :3])
    
    projected_points = points[:, :2]  # 仅保留x,y坐标
    return projected_points, colors

def parse_annotation(txt_path, room_id):
    """解析TXT标注文件，返回实体列表（增加room_id标识）"""
    entities = {"walls": [], "doors": [], "windows": [], "bboxes": []}
    wall_id_map = {}  # 用于Door/Window关联Wall
    
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in lines:
        # 匹配Wall
        wall_match = re.match(r"wall_(\d+)=Wall\(([^)]+)\)", line)
        if wall_match:
            wall_id = int(wall_match.group(1))
            params = list(map(float, wall_match.group(2).split(",")))
            if len(params) != 8:
                print(f"跳过无效Wall格式: {line}")
                continue
            ax, ay, az, bx, by, bz, height, thickness = params
            wall = Wall(
                id=wall_id,
                room_id=room_id,
                ax=ax, ay=ay, az=az,
                bx=bx, by=by, bz=bz,
                height=height, thickness=thickness
            )
            entities["walls"].append(wall)
            wall_id_map[wall_id] = wall
            continue
        
        # 匹配Door
        door_match = re.match(r"door_(\d+)=Door\(wall_(\d+),([^)]+)\)", line)
        if door_match:
            door_id = int(door_match.group(1))
            wall_id = int(door_match.group(2))
            params = list(map(float, door_match.group(3).split(",")))
            if len(params) != 5 or wall_id not in wall_id_map:
                print(f"跳过无效Door格式: {line}")
                continue
            x, y, z, width, height = params
            door = Door(
                id=door_id,
                room_id=room_id,
                wall_id=wall_id,
                position_x=x, position_y=y, position_z=z,
                width=width, height=height
            )
            entities["doors"].append(door)
            continue
        
        # 匹配Window
        window_match = re.match(r"window_(\d+)=Window\(wall_(\d+),([^)]+)\)", line)
        if window_match:
            window_id = int(window_match.group(1))
            wall_id = int(window_match.group(2))
            params = list(map(float, window_match.group(3).split(",")))
            if len(params) != 5 or wall_id not in wall_id_map:
                print(f"跳过无效Window格式: {line}")
                continue
            x, y, z, width, height = params
            window = Window(
                id=window_id,
                room_id=room_id,
                wall_id=wall_id,
                position_x=x, position_y=y, position_z=z,
                width=width, height=height
            )
            entities["windows"].append(window)
            continue
        
        # 匹配Bbox
        bbox_match = re.match(r"bbox_(\d+)=Bbox\(([^,]+),([^)]+)\)", line)
        if bbox_match:
            bbox_id = int(bbox_match.group(1))
            class_name = bbox_match.group(2)
            params = list(map(float, bbox_match.group(3).split(",")))
            if len(params) != 7:
                print(f"跳过无效Bbox格式: {line}")
                continue
            x, y, z, angle_z, scale_x, scale_y, scale_z = params
            bbox = Bbox(
                id=bbox_id,
                room_id=room_id,
                class_name=class_name,
                position_x=x, position_y=y, position_z=z,
                angle_z=angle_z, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z
            )
            entities["bboxes"].append(bbox)
            continue
    
    return entities

def get_bbox_corners_2d(bbox):
    """计算Bbox在XY平面上的旋转后矩形角点"""
    center = np.array([bbox.position_x, bbox.position_y])
    half_w = bbox.scale_x / 2.0
    half_h = bbox.scale_y / 2.0
    corners = np.array([
        [half_w, half_h],
        [half_w, -half_h],
        [-half_w, -half_h],
        [-half_w, half_h]
    ])
    rot_mat = R.from_rotvec([0, 0, bbox.angle_z]).as_matrix()[:2, :2]
    rotated_corners = (rot_mat @ corners.T).T + center
    return rotated_corners

def get_wall_angle(wall):
    """计算Wall在XY平面上的方向角度"""
    dir_x = wall.bx - wall.ax
    dir_y = wall.by - wall.ay
    angle = np.arctan2(dir_y, dir_x)
    return angle

def get_door_window_corners_2d(door_window, wall):
    """计算Door/Window在XY平面上的旋转后矩形角点"""
    center = np.array([door_window.position_x, door_window.position_y])
    half_width = door_window.width / 2.0
    half_thickness = 0.05
    corners = np.array([
        [half_width, half_thickness],
        [half_width, -half_thickness],
        [-half_width, -half_thickness],
        [-half_width, half_thickness]
    ])
    wall_angle = get_wall_angle(wall)
    rot_mat = R.from_rotvec([0, 0, wall_angle]).as_matrix()[:2, :2]
    rotated_corners = (rot_mat @ corners.T).T + center
    return rotated_corners

# --------------------------
# 3. 可视化与主函数
# --------------------------
def visualize_2d_with_annotations(all_points, all_colors, all_entities, scene_name, save_path):
    """绘制包含所有房间的2D投影图"""
    plt.figure(figsize=(16, 14))
    ax = plt.gca()

    ax.set_aspect('equal', adjustable='box')  # 保证x和y方向的刻度单位长度一致
    
    # 1. 绘制所有房间的投影点云
    for i, (points, colors) in enumerate(zip(all_points, all_colors)):
        ax.scatter(points[:, 0], points[:, 1], 
                  c=colors[:, :3], s=1, alpha=0.6, 
                  label=f"Room {i} Point Cloud" if i == 0 else "")
    
    # 2. 绘制所有Wall
    for wall in all_entities["walls"]:
        ax.plot([wall.ax, wall.bx], [wall.ay, wall.by], 
                color="#FF5733", linewidth=3, label="Wall" if wall.id == 0 and wall.room_id == 0 else "")
        mid_x = (wall.ax + wall.bx) / 2
        mid_y = (wall.ay + wall.by) / 2
        ax.text(mid_x, mid_y, f"R{wall.room_id}_W{wall.id}", fontsize=8, 
                ha="center", va="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # 3. 绘制所有Door
    for door in all_entities["doors"]:
        wall = next((w for w in all_entities["walls"] if w.id == door.wall_id and w.room_id == door.room_id), None)
        if not wall:
            print(f"Door_{door.id} (Room {door.room_id}) 未找到关联的Wall，跳过绘制")
            continue
        corners = get_door_window_corners_2d(door, wall)
        corners = np.vstack([corners, corners[0]])
        ax.plot(corners[:, 0], corners[:, 1], color="#2E8B57", linewidth=2, 
                label="Door" if door.id == 0 and door.room_id == 0 else "")
        ax.text(door.position_x, door.position_y, f"R{door.room_id}_D{door.id}", fontsize=8, 
                ha="center", va="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 4. 绘制所有Window
    for window in all_entities["windows"]:
        wall = next((w for w in all_entities["walls"] if w.id == window.wall_id and w.room_id == window.room_id), None)
        if not wall:
            print(f"Window_{window.id} (Room {window.room_id}) 未找到关联的Wall，跳过绘制")
            continue
        corners = get_door_window_corners_2d(window, wall)
        corners = np.vstack([corners, corners[0]])
        ax.plot(corners[:, 0], corners[:, 1], color="#FFA500", linewidth=2, 
                label="Window" if window.id == 0 and window.room_id == 0 else "")
        ax.text(window.position_x, window.position_y, f"R{window.room_id}_Win{window.id}", fontsize=8, 
                ha="center", va="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 5. 绘制所有Bbox
    for bbox in all_entities["bboxes"]:
        corners = get_bbox_corners_2d(bbox)
        corners = np.vstack([corners, corners[0]])
        ax.plot(corners[:, 0], corners[:, 1], color="#32CD32", linewidth=2, 
                label="Bbox" if bbox.id == 0 and bbox.room_id == 0 else "")
        ax.text(bbox.position_x, bbox.position_y, f"R{bbox.room_id}_{bbox.class_name}", fontsize=8, 
                ha="center", va="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # 设置图表属性
    ax.set_xlabel("X Coordinate (m)", fontsize=12)
    ax.set_ylabel("Y Coordinate (m)", fontsize=12)
    ax.set_title(f"2D Projection of {scene_name}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"包含所有房间的2D图已保存至: {save_path}")

    # 强制设置坐标提示格式，避免 Unicode 异常
    ax.format_coord = lambda x, y: f"x={x:.2f} m, y={y:.2f} m"

    #交互窗口
    plt.show()  # 打开交互窗口，可缩放、平移查看细节
    plt.close()



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PLY 2D-Projection with TXT Annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 位置参数
    parser.add_argument("output_path", type=Path, help="输出目录路径")
    parser.add_argument("ply_dir", type=Path, help="PLY文件所在目录")
    parser.add_argument("txt_dir", type=Path, help="TXT文件所在目录")
    parser.add_argument("scene_id", help="场景ID (如: scene_0001)")
    
    # 样本ID（必需）
    parser.add_argument("--sample", "-s", type=int, required=True, help="样本ID (如: 0, 1, 2...)")
    
    # 房间选择（互斥）
    room_group = parser.add_mutually_exclusive_group(required=True)
    room_group.add_argument("--room", "-r", type=str, help="房间ID (支持前导零，如: 00, 01)")
    room_group.add_argument("--all", "-a", action="store_true", help="处理所有房间")
    
    args = parser.parse_args()
    
    # 基础验证（仅验证目录存在）
    if not args.ply_dir.is_dir():
        parser.error(f"PLY目录不存在: {args.ply_dir}")
    if not args.txt_dir.is_dir():
        parser.error(f"TXT目录不存在: {args.txt_dir}")
    
    # 确保输出目录存在
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建文件匹配模式
    room_pattern = "*" if args.all else args.room
    pattern = f"{args.scene_id}_{room_pattern}_{args.sample}.ply"
    
    # 查找文件
    ply_files = sorted(args.ply_dir.glob(pattern))
    if not ply_files:
        parser.error(f"未找到匹配 '{pattern}' 的文件")
    
    # 构建文件对（假设文件名格式严格规范）
    ply_txt_pairs = []
    valid_rooms = []
    
    for ply_path in ply_files:
        # 直接解析文件名: scene_{id}_{room}_{sample}.ply
        prefix, file_scene_id, room_id, sample_id = ply_path.stem.split('_')
        
        # 查找同名txt文件（假设一定存在）
        txt_path = args.txt_dir / f"{ply_path.stem}.txt"
        
        ply_txt_pairs.append((ply_path, txt_path, room_id))
        valid_rooms.append(room_id)
    
    # 生成输出信息
    if args.all:
        scene_name = f"{args.scene_id} (Sample {args.sample}) - All Rooms [{len(valid_rooms)} rooms]"
        save_filename = f"{args.scene_id}_sample_{args.sample}_all_rooms.png"
        print(f"找到 {len(valid_rooms)} 个房间: {sorted(valid_rooms)}")
    else:
        scene_name = f"{args.scene_id} (Sample {args.sample}) - Room {args.room}"
        save_filename = f"{args.scene_id}_sample_{args.sample}_room_{args.room}.png"
    
    save_path = args.output_path / save_filename
    
    return scene_name, save_path, ply_txt_pairs


def main(scene_name, save_path, ply_txt_room_pairs):
    """
    ply_txt_room_pairs: [(ply_path, txt_path, room_id), ...]
    """
    all_points, all_colors, all_entities = [], [], {"walls": [], "doors": [], "windows": [], "bboxes": []}
    
    # 按room_id排序
    ply_txt_room_pairs.sort(key=lambda x: x[2])
    
    for ply_path, txt_path, room_id in ply_txt_room_pairs:
        print(f"  [Room {room_id}] {ply_path.name}")
        
        # 读取并解析
        projected_points, colors = read_ply_and_project(ply_path)
        entities = parse_annotation(str(txt_path), room_id)
        
        all_points.append(projected_points)
        all_colors.append(colors)
        all_entities["walls"].extend(entities["walls"])
        all_entities["doors"].extend(entities["doors"])
        all_entities["windows"].extend(entities["windows"])
        all_entities["bboxes"].extend(entities["bboxes"])
    
    visualize_2d_with_annotations(all_points, all_colors, all_entities, scene_name, str(save_path))


if __name__ == "__main__":
    scene_name, save_path, ply_txt_pairs = parse_arguments()
    main(scene_name, save_path, ply_txt_pairs)