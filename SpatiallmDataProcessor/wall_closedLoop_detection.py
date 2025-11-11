#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Contact       : ZhangJie
# @Date         : 2025-10-30
# @Version      : 0.0.1
"""
====================================================================
Wall 闭合多边形检测工具 (Wall Closed Loop Detection Tool)
====================================================================
功能描述：
批量检测文件夹中所有 txt 标注文件内的 wall 实体是否能构成闭合多边形。
支持两种检测模式：
1. 有向边检测：验证 wall 首尾端点是否按顺序严格连接（A→B 与 B→C 相连）
2. 无向边检测：基于图论判断 wall 作为无向边是否形成单环（各点度数为2且连通）

核心流程：
1. 从 txt 文件中提取所有以 "wall_" 开头的标注行
2. 解析为 Wall 实体对象（包含端点坐标等属性）
3. 尝试对 wall 进行排序以形成环状结构
4. 检测排序后的 wall 是否构成闭合多边形
5. 汇总所有文件的检测结果并输出日志

使用说明：
1. 修改脚本末尾的 TXT_FOLDER_PATH 为目标文件夹路径
2. 运行脚本，自动处理该文件夹下所有 .txt 文件
3. 结果会在控制台显示，并生成失败文件日志（wall_closedLoop_detection_failed_files.log）

输入文件格式要求：
txt 文件中需包含形如 "wall_0=Wall(ax, ay, az, bx, by, bz, height, thickness)" 的标注行
其中 ax, ay, az 为起点坐标，bx, by, bz 为终点坐标

输出说明：
- 控制台显示每个文件的处理结果（成功/失败）
- 汇总统计总文件数、成功闭合数、失败数
- 失败文件列表写入日志文件，便于后续检查
--------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import os
import re

from entity import Wall

# ====================== 工具函数 ======================
# 工具函数：解析 wall_ 开头的行
def parse_wall_line(line: str):
    pattern = r'^(\w+)=(\w+)\((.*)\)$'
    match = re.match(pattern, line.strip())
    if not match:
        return None
    name, cls_name, params = match.groups()
    if cls_name != 'Wall':
        return None

    params_list = [p.strip() for p in params.split(',')]
    if len(params_list) != 8:
        raise ValueError(f"Wall 参数数量不对，期望 8 个，实际 {len(params_list)}: {params}")

    try:
        wall = Wall(
            id=name.split('_')[1],  # 如 wall_0 → '0'
            ax=float(params_list[0]),
            ay=float(params_list[1]),
            az=float(params_list[2]),
            bx=float(params_list[3]),
            by=float(params_list[4]),
            bz=float(params_list[5]),
            height=float(params_list[6]),
            thickness=float(params_list[7])
        )
        return wall
    except Exception as e:
        print(f"解析 Wall 失败: {e}，行内容: {line}")
        return None

# 工具函数：提取某个 txt 文件中所有 wall_ 开头的行
def extract_wall_lines_from_txt(file_path: str) -> List[str]:
    wall_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('wall_'):
                wall_lines.append(line)
    return wall_lines

# 工具函数：将 wall_ 行列表解析为 List[Wall]
def parse_walls_from_lines(wall_lines: List[str]) -> List[Wall]:
    walls = []
    for line in wall_lines:
        wall = parse_wall_line(line)
        if wall:
            walls.append(wall)
    return walls

# ====================== 排序函数 ======================
# 有向walls排序函数
def sort_walls_to_form_ring(walls: List[Wall]) -> Optional[List[Wall]]:
    if not walls:
        return None
    if len(walls) == 1:
        return walls  # 1个wall无法成环，但返回原样

    n = len(walls)
    used = [False] * n

    def try_build_chain(start_idx: int) -> Optional[List[Wall]]:
        sorted_walls = []
        used_copy = used.copy()
        current_idx = start_idx
        sorted_walls.append(walls[current_idx])
        used_copy[current_idx] = True

        for _ in range(n - 1):
            last_wall = sorted_walls[-1]
            found_next = False
            next_idx = -1

            for i in range(n):
                if not used_copy[i]:
                    candidate = walls[i]
                    if (last_wall.bx, last_wall.by) == (candidate.ax, candidate.ay):
                        next_idx = i
                        found_next = True
                        break

            if not found_next:
                return None  # 无法找到下一个匹配的 wall

            sorted_walls.append(walls[next_idx])
            used_copy[next_idx] = True

        # 检查是否首尾相连成环
        first_wall = sorted_walls[0]
        last_wall = sorted_walls[-1]
        if (last_wall.bx, last_wall.by) == (first_wall.ax, first_wall.ay):
            return sorted_walls
        else:
            return None

    # 尝试以每个 wall 为起点构建环
    for start_idx in range(n):
        sorted_result = try_build_chain(start_idx)
        if sorted_result:
            return sorted_result

    # 所有起点尝试都失败
    return None

# 无向walls排序函数(TODO)
def sort_walls_as_undirected_edges_to_cycle(walls: List[Wall]) -> Optional[List[Wall]]:
    pass
    # return None  # 所有尝试都失败


# ====================== wall成环检测 ======================
# 判断有向 walls 是否首尾相连成环（闭合多边形）
def is_wall_closed_polygon(walls: List[Wall]) -> bool:
    n = len(walls)
    if n < 3:
        print(f"⚠️  Wall 数量不足 3 个，无法构成多边形，当前 walls 数量: {n}")
        return False

    for i in range(n):
        curr_wall = walls[i]
        next_wall = walls[(i + 1) % n]

        curr_end = (curr_wall.bx, curr_wall.by)
        next_start = (next_wall.ax, next_wall.ay)

        if curr_end != next_start:
            print(f"❌ Wall {i} 的终点 {curr_end} != Wall {(i+1)%n} 的起点 {next_start}")
            return False

    print(f"✅ 成功：{n} 个 walls 首尾相连，构成闭合多边形！")
    return True

# 基于图论的无向边环检测
def is_unoriented_wall_cycle(walls: List[Wall]) -> bool:
    if len(walls) < 3:
        return False

    # Step 1: 构建图：点 -> 相邻点列表，并统计度数
    graph: Dict[Tuple[float, float], List[Tuple[float, float]]] = defaultdict(list)
    degrees: Dict[Tuple[float, float], int] = defaultdict(int)
    points = set()

    for w in walls:
        a = (w.ax, w.ay)
        b = (w.bx, w.by)
        points.add(a)
        points.add(b)
        graph[a].append(b)
        graph[b].append(a)
        degrees[a] += 1
        degrees[b] += 1

    # Step 2: 检查每个点的度数是否 == 2
    for p, d in degrees.items():
        if d != 2:
            return False

    # Step 3: 检查是否连通（任意一点出发，是否能访问所有点）
    if not points:
        return False

    start = next(iter(points))
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        p = queue.popleft()
        for neighbor in graph[p]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # 所有点都应该被访问到
    return len(visited) == len(points)


# ====================== 主函数：批量处理某个文件夹中的所有 txt 文件 ======================
# -------------------- 全局汇总变量 --------------------
TOTAL_FILES = 0
SUCCESS_FILES = []
FAILED_FILES  = []
def batch_process_txt_files(folder_path: str):
    global TOTAL_FILES, SUCCESS_FILES, FAILED_FILES
    if not os.path.isdir(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"⚠️  文件夹 {folder_path} 中没有 .txt 文件")
        return

    print(f"🔍 开始处理文件夹: {folder_path}，共发现 {len(txt_files)} 个 txt 文件")

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        print(f"\n📄 正在处理文件: {txt_file}")

        # Step 1: 提取所有 wall_ 开头的行
        wall_lines = extract_wall_lines_from_txt(file_path)
        if not wall_lines:
            print(f"  ⚠️  文件 {txt_file} 中没有找到任何 wall_ 开头的行")
            continue

        print(f"  🧱 提取到 {len(wall_lines)} 个 wall_ 行")

        # Step 2: 解析为 List[Wall]
        walls = parse_walls_from_lines(wall_lines)
        if not walls:
            print(f"  ❌ 文件 {txt_file} 中没有成功解析出任何 Wall 对象")
            continue
        print(f"  ✅ 成功解析出 {len(walls)} 个 Wall 对象")


        # Step 3: 尝试排序 walls 成环
        sorted_walls = walls
        # closed = is_wall_closed_polygon(walls) # 旧：顺序匹配 wall(0-> 1-> 2 ->..-> 0）
        # sorted_walls = sort_walls_to_form_ring(walls)  # 旧：有向边匹配
        # sorted_walls = sort_walls_as_undirected_edges_to_cycle(walls)  # 新：无向边匹配

        # Step 4: 判断是否构成闭合多边形
        if sorted_walls:
            print(f"  🔁 成功对 {len(sorted_walls)} 个 walls 自动排序，形成可能的环状序列")
            # closed = is_wall_closed_polygon(sorted_walls)
            closed = is_unoriented_wall_cycle(sorted_walls)
        else:
            print(f"  ❌ 无法将 {len(walls)} 个 walls 排序成首尾相连的环")
            closed = False

        # 记录结果
        if closed:
            SUCCESS_FILES.append(txt_file)
        else:
            FAILED_FILES.append(txt_file)

        # Step 4: 输出结果
        print(f"  🎯 文件 {txt_file} 最终判定：是否闭合多边形？ {'是' if closed else '否'}")

    # -------------------- 最终汇总 --------------------
    print("=" * 60)
    print("📊 处理完毕，汇总如下：")
    print(f"   总检测文件数 : {TOTAL_FILES}")
    print(f"   成功闭合数   : {len(SUCCESS_FILES)}")
    print(f"   失败/异常数  : {len(FAILED_FILES)}")
    if FAILED_FILES:
        print("   失败文件列表 :")
        for name in FAILED_FILES:
            print(f"     - {name}")
    print("=" * 60)

    # 可选：把失败文件名落盘
    with open("wall_closedLoop_detection_failed_files.log", "w", encoding="utf-8") as log:
        log.write("\n".join(FAILED_FILES))
    print("\n📝 失败文件名已写入 failed_files.log")

if __name__ == '__main__':
    # 📂 替换 txt 文件所在的目录
    # TXT_FOLDER_PATH = '../txt'  # chunk1的全部txt
    TXT_FOLDER_PATH = './txt_predict'  # 
    batch_process_txt_files(TXT_FOLDER_PATH)