#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Contact       : ZhangJie
# @ Date         : 2025-11-03
# @ Version      : 0.0.1
"""
====================================================================
Wall Polygon vs BBox 2D-Overlap Batch Analyzer
====================================================================
介绍：
批量解析场景文件夹下所有 scene_*.txt，重建墙体多边形，并计算每个
bounding-box 与墙多边形之间的 2D 重叠/超出关系，最终汇总为单 CSV。

核心指标：
- fully_inside_no_touch : BBox 完全在墙内（不含边界）
- inside_or_on_boundary : BBox 在墙内或边界上
- practically_inside    : 上述二者之一，或超出面积 < 1e-4 m²
- overlap_ratio         : 超出墙面积 / BBox 总面积
- intersection_ratio    : 与墙相交面积 / BBox 总面积

输入：
单个文件夹路径，内含任意数量 scene_*.txt（每文件含墙体与BBox定义）

输出：
1. 汇总 CSV（含 source_file 字段区分不同场景）
2. 可选：每场景可视化 PNG（*_visualization.png，红区=超出墙范围）
--------------------------------------------------------------------
命令行用法

"""

import sys
import os
import math
import csv
import logging
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate
from shapely.validation import explain_validity
from SpatiallmDataProcessor.utils.entity import Wall, Bbox


# 配置日志：输出到文件，仅记录必要信息
logging.basicConfig(
    filename='out_bbox_out_ratio/.log',
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    filemode='w'
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# 定义微小面积阈值，如果面积小于此阈值则认为没有超出墙范围，practically_inside=True
MIN_OUTSIDE_AREA = 1e-4  # 0.0001 m²

# 可视化修复后的墙体多边形
VISUALIZE_POLYGON_REPAIR=False
# VISUALIZE_POLYGON_REPAIR=True

# 可视化墙体与BBox的重叠关系
VISUALIZE_WALL_BBOX_OVERLAP=False
# VISUALIZE_WALL_BBOX_OVERLAP=True

# 获取 walls 和 bbox 列表
def parse_scene_file(filename):
    walls = []
    bboxes = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('wall_'):
                parts = line.split('=')
                wall_id = int(parts[0].split('_')[1])
                wall_data = parts[1].replace('Wall(', '').rstrip(')').split(',')
                wall_data = [float(x.strip()) for x in wall_data]
                
                wall = Wall(
                    id=wall_id,
                    ax=wall_data[0],
                    ay=wall_data[1],
                    az=wall_data[2],
                    bx=wall_data[3],
                    by=wall_data[4],
                    bz=wall_data[5],
                    height=wall_data[6],
                    thickness=wall_data[7]
                )
                walls.append(wall)
            
            elif line.startswith('bbox_'):
                parts = line.split('=')
                bbox_id = int(parts[0].split('_')[1])
                bbox_data = parts[1].replace('Bbox(', '').rstrip(')').split(',')
                
                class_name = bbox_data[0].strip()
                numeric_data = [float(x.strip()) for x in bbox_data[1:]]
                
                bbox = Bbox(
                    id=bbox_id,
                    class_name=class_name,
                    position_x=numeric_data[0],
                    position_y=numeric_data[1],
                    position_z=numeric_data[2],
                    angle_z=numeric_data[3],
                    scale_x=numeric_data[4],
                    scale_y=numeric_data[5],
                    scale_z=numeric_data[6]
                )
                bboxes.append(bbox)
    
    walls.sort(key=lambda x: x.id)
    bboxes.sort(key=lambda x: x.id)
    return walls, bboxes



def create_wall_polygon(walls):
    if not walls:
        print("Error[line 101]: No walls found")
        return None
        
    polygon_points = []
    for wall in walls:
        polygon_points.append((wall.ax, wall.ay))
    
    if polygon_points and polygon_points[0] != polygon_points[-1]:
        polygon_points.append(polygon_points[0])
        
    return Polygon(polygon_points) 



def get_bbox_polygon(bbox):
    half_x = bbox.scale_x / 2.0
    half_y = bbox.scale_y / 2.0
    
    corners = [
        (bbox.position_x - half_x, bbox.position_y - half_y),
        (bbox.position_x + half_x, bbox.position_y - half_y),
        (bbox.position_x + half_x, bbox.position_y + half_y),
        (bbox.position_x - half_x, bbox.position_y + half_y)
    ]
    
    bbox_poly = Polygon(corners)
    # 正角度=逆时针
    return rotate(bbox_poly, math.degrees(bbox.angle_z), origin='center')


def calculate_overlap_ratio(bbox_poly, wall_poly):
    # 保留原始计算结果，不做四舍五入
    intersection = bbox_poly.intersection(wall_poly)
    outside_part = bbox_poly.difference(wall_poly)
    intersection_area = intersection.area
    bbox_area = bbox_poly.area
    outside_area = outside_part.area if not outside_part.is_empty else 0.0

    if bbox_area == 0:
        return 0.0, bbox_poly, outside_part, outside_area
    # 直接计算原始比例（不四舍五入）
    overlap_ratio = 1.0 - (intersection_area / bbox_area)
    return overlap_ratio, bbox_poly, outside_part, outside_area  # 新增返回原始outside_area


# 可视化函数
def visualize_wall_bbox_overlap(walls, bboxes, wall_poly, title_suffix=""):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # 绘制墙体多边形
    if wall_poly and wall_poly.is_valid:
        wall_x, wall_y = wall_poly.exterior.xy
        ax.plot(wall_x, wall_y, 'b-', linewidth=2, label='Wall Polygon')
        ax.fill(wall_x, wall_y, 'blue', alpha=0.1)

    # 绘制边界框及超出部分
    colors = ['#FFA500', '#228B22', '#9370DB', '#FF6347', '#1E90FF']
    for idx, bbox in enumerate(bboxes):
        bbox_poly = get_bbox_polygon(bbox)
        _, _, outside_part, _ = calculate_overlap_ratio(bbox_poly, wall_poly)
        
        bbox_x, bbox_y = bbox_poly.exterior.xy
        color = colors[idx % len(colors)]
        ax.plot(bbox_x, bbox_y, color=color, linewidth=1.5, 
                label=f'Bbox_{bbox.id} ({bbox.class_name})')
        
        if not outside_part.is_empty:
            if outside_part.geom_type == 'MultiPolygon':
                for part in outside_part.geoms:
                    out_x, out_y = part.exterior.xy
                    ax.fill(out_x, out_y, 'red', alpha=0.5)
            else:
                out_x, out_y = outside_part.exterior.xy
                ax.fill(out_x, out_y, 'red', alpha=0.5)
        
        centroid = bbox_poly.centroid
        ax.text(centroid.x, centroid.y, f'ID:{bbox.id}', 
                fontsize=8, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title(f'2D Projection {title_suffix} (Red = Outside Wall)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # 强制设置坐标提示格式，避免 Unicode 异常
    ax.format_coord = lambda x, y: f"x={x:.2f} m, y={y:.2f} m"
    # 保存图片
    plt.savefig(f"{title_suffix}_visualization.png", dpi=300)
    # 显示图像
    plt.show()


# 分析 wall 构成的闭合多边形是否有效，并分析无效的原因
def analyze_and_log_polygon_issues(source_filename, wall_polygon, walls):
    """
    专门分析多边形无效原因并记录详细日志
    :param source_filename: 源文件名（用于定位问题）
    :param wall_polygon: 待检测的多边形对象（可能为None）
    :param walls: 原始墙体数据（用于提取顶点和辅助分析）
    :return: bool - 多边形是否有效（True为有效，False为无效）
    """
    # 情况1：多边形未创建（无墙体数据）
    if wall_polygon is None:
        error_msg = f"{source_filename} - 多边形创建失败：无墙体数据（walls为空）"
        logging.warning(error_msg)
        print(f"警告：{error_msg}")
        return False

    # 情况2：多边形有效，无需处理
    if wall_polygon.is_valid:
        return True

    # 情况3：多边形无效，详细分析原因
    invalid_reason = explain_validity(wall_polygon)
    error_details = []

    # 提取原始顶点（与create_wall_polygon逻辑一致）
    raw_points = [(wall.ax, wall.ay) for wall in walls] if walls else []
    if raw_points and raw_points[0] != raw_points[-1]:
        raw_points.append(raw_points[0])  # 模拟闭合处理
    point_count = len(raw_points)

    # 子情况3.1：顶点数量不足
    if point_count < 3:
        error_details.append(f"顶点数量不足（仅{point_count}个，至少需要3个）")

    # 子情况3.2：存在连续共线顶点
    def has_continuous_collinear(points):
        """检测是否存在连续3点共线"""
        if len(points) < 3:
            return False
        for i in range(len(points) - 2):
            p1, p2, p3 = points[i], points[i+1], points[i+2]
            # 向量叉积判断共线（值接近0视为共线）
            cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if abs(cross_product) < 1e-9:
                return True
        return False
    if has_continuous_collinear(raw_points):
        error_details.append("存在连续共线顶点（可能导致拓扑无效）")

    # 子情况3.3：自相交（交叉或形成环）
    if "Self-intersection" in invalid_reason:
        # 提取自相交位置（从错误信息中解析）
        intersect_pos = invalid_reason.split("at ")[-1].strip() if "at " in invalid_reason else "未知位置"
        error_details.append(f"自相交（交叉点：{intersect_pos}）")

    # 子情况3.4：多边形未闭合（极端浮点误差导致）
    if not wall_polygon.exterior.is_closed:
        error_details.append("多边形未闭合（首尾顶点不重合）")

    # 子情况3.5：其他未归类原因
    if not error_details:
        error_details.append(f"未归类错误：{invalid_reason}")

    # 汇总日志信息
    full_error_msg = (
        f"{source_filename} - 多边形无效（顶点数：{point_count}）："
        f"{'; '.join(error_details)}"
    )
    logging.warning(full_error_msg)
    print(f"警告：{full_error_msg}")
    return False


# 尝试修复无效多边形，返回修复后的多边形或None（修复失败）
def repair_invalid_polygon(source_filename, wall_polygon, walls):
    """
    尝试修复无效多边形，返回修复后的多边形或None（修复失败）
    :param source_filename: 源文件名
    :param wall_polygon: 原始无效多边形
    :param walls: 原始墙体数据（用于提取顶点）
    :return: 修复后的有效多边形或None
    """
    if wall_polygon is None:
        logging.warning(f"{source_filename} - 无多边形可修复（原始多边形为None）")
        return None

    # 提取原始顶点（与create_wall_polygon逻辑一致）
    raw_points = [(wall.ax, wall.ay) for wall in walls] if walls else []
    if raw_points and raw_points[0] != raw_points[-1]:
        raw_points.append(raw_points[0])

    # 步骤1：处理顶点数量不足（若有足够墙体数据，尝试补全）
    if len(raw_points) < 3:
        logging.warning(f"{source_filename} - 顶点不足，无法修复（需至少3个顶点）")
        return None

    # 步骤2：移除连续共线顶点（解决共线导致的拓扑无效）
    def remove_collinear_points(points):
        """移除连续共线的冗余顶点"""
        if len(points) <= 3:
            return points  # 不足3点无需处理
        filtered = [points[0]]  # 保留第一个点
        for i in range(1, len(points)-1):
            p1, p2, p3 = filtered[-1], points[i], points[i+1]
            # 向量叉积判断共线
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            if abs(cross) >= 1e-9:  # 非共线则保留
                filtered.append(p2)
        filtered.append(points[-1])  # 保留最后一个点
        # 确保闭合
        if filtered[0] != filtered[-1]:
            filtered.append(filtered[0])
        return filtered

    # 检测是否存在共线问题
    has_collinear = any(
        abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])) < 1e-9
        for i, (p1, p2, p3) in enumerate(zip(raw_points[:-2], raw_points[1:-1], raw_points[2:]))
    )
    if has_collinear:
        repaired_points = remove_collinear_points(raw_points)
        wall_polygon = Polygon(repaired_points)
        logging.info(f"{source_filename} - 已移除共线顶点（原始{len(raw_points)}个→修复后{len(repaired_points)}个）")
        if wall_polygon.is_valid:
            return wall_polygon

    # 步骤3：处理自相交（用buffer(0)修复常见自相交问题）
    invalid_reason = explain_validity(wall_polygon)
    if "Self-intersection" in invalid_reason:
        try:
            repaired_poly = wall_polygon.buffer(0)
            # 当修复结果为多多边形时，选择面积最大的部分
            if repaired_poly.geom_type == "MultiPolygon":
                # 按面积降序排序并选择最大的多边形
                largest_poly = max(repaired_poly.geoms, key=lambda g: g.area)
                repaired_poly = largest_poly
                logging.info(f"{source_filename} - 自相交修复后选择最大子多边形（面积：{largest_poly.area:.4f}）")
            
            if repaired_poly.is_valid and repaired_poly.geom_type == "Polygon":
                logging.info(f"{source_filename} - 已修复自相交问题（{invalid_reason}）")
                return repaired_poly
            else:
                logging.warning(f"{source_filename} - 自相交修复后仍无效（类型：{repaired_poly.geom_type}）")
        except Exception as e:
            logging.error(f"{source_filename} - 自相交修复失败：{str(e)}")

    # 步骤4：处理未闭合问题（强制闭合）
    if not wall_polygon.exterior.is_closed:
        closed_points = list(wall_polygon.exterior.coords)
        if closed_points and closed_points[0] != closed_points[-1]:
            closed_points.append(closed_points[0])
            repaired_poly = Polygon(closed_points)
            if repaired_poly.is_valid:
                logging.info(f"{source_filename} - 已修复未闭合问题")
                return repaired_poly

    # 所有修复尝试失败
    logging.error(f"{source_filename} - 所有修复尝试失败（最终原因：{explain_validity(wall_polygon)}）")
    return None


def visualize_polygon_repair(source_filename, original_poly, repaired_poly, raw_points):
    """
    可视化原始多边形与修复后的多边形对比
    :param source_filename: 文件名（用于标题）
    :param original_poly: 原始无效多边形
    :param repaired_poly: 修复后的多边形（可能为None）
    :param raw_points: 原始顶点列表（用于标记点）
    """
    # 设置画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"多边形修复对比 - {source_filename}", fontsize=14)

    # ---------------------- 绘制原始多边形 ----------------------
    ax1.set_title("原始多边形（无效）")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_aspect('equal')  # 等比例显示

    # 绘制原始顶点（红色点）
    if raw_points:
        x, y = zip(*raw_points)
        ax1.scatter(x, y, c='red', s=50, label='原始顶点')

    # 绘制原始多边形轮廓（红色虚线，可能混乱）
    if original_poly:
        try:
            # 提取边界坐标（排除最后一个重复的闭合点）
            coords = list(original_poly.exterior.coords[:-1])
            if coords:
                x_poly, y_poly = zip(*coords)
                ax1.plot(x_poly + (x_poly[0],), y_poly + (y_poly[0],), 'r--', label='原始轮廓')
        except Exception as e:
            ax1.text(0.5, 0.5, f"绘制失败: {str(e)}", ha='center', va='center')

    ax1.legend()

    # ---------------------- 绘制修复后多边形 ----------------------
    ax2.set_title("修复后多边形（有效/无效）")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_aspect('equal')

    # 绘制原始顶点（灰色点，作为参考）
    if raw_points:
        x, y = zip(*raw_points)
        ax2.scatter(x, y, c='gray', s=30, alpha=0.5, label='原始顶点（参考）')

    # 绘制修复后多边形
    if repaired_poly:
        # 修复后的顶点（蓝色点）
        repaired_coords = list(repaired_poly.exterior.coords[:-1])
        if repaired_coords:
            x_repair, y_repair = zip(*repaired_coords)
            ax2.scatter(x_repair, y_repair, c='blue', s=50, label='修复后顶点')

            # 修复后的轮廓（蓝色实线）
            ax2.plot(x_repair + (x_repair[0],), y_repair + (y_repair[0],), 'b-', label='修复后轮廓')

        # 标记是否有效
        validity = "有效" if repaired_poly.is_valid else "无效"
        ax2.text(0.05, 0.95, f"状态: {validity}", transform=ax2.transAxes, 
                 bbox=dict(facecolor='green' if validity == "有效" else 'red', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, "修复失败（无结果）", ha='center', va='center',
                 bbox=dict(facecolor='red', alpha=0.5))

    ax2.legend()
    plt.tight_layout()
    plt.show()


def process_file(input_file=False):
    """
    处理单个txt文件，返回带来源文件名的bbox结果列表
    :param input_file: 单个txt文件路径
    :param visualize_repair: 是否可视化修复前后对比
    :param visualize_bbox_overlaps: 是否可视化bbox与墙体的重叠关系
    :return: 含source_file的bbox结果列表（空列表表示处理失败）
    """
    source_filename = os.path.basename(input_file)
    walls, bboxes = parse_scene_file(input_file)  # 假设已定义
    original_poly = create_wall_polygon(walls)    # 原始构建的多边形

    # 提取原始顶点（用于可视化）
    raw_points = [(wall.ax, wall.ay) for wall in walls] if walls else []
    if raw_points and raw_points[0] != raw_points[-1]:
        raw_points.append(raw_points[0])  # 与创建逻辑一致的闭合点

    is_valid = analyze_and_log_polygon_issues(source_filename, original_poly, walls)
    repaired_poly = original_poly if is_valid else None

    # 对于无效多边形，尝试修复
    if not is_valid:
        repaired_poly = repair_invalid_polygon(source_filename, original_poly, walls)
        if VISUALIZE_POLYGON_REPAIR:
            visualize_polygon_repair(source_filename, original_poly, repaired_poly, raw_points)

        if repaired_poly is None:
            # 修复失败，若需要可视化失败案例
            if VISUALIZE_POLYGON_REPAIR:
                visualize_polygon_repair(source_filename, original_poly, None, raw_points)
            return []
        # 修复后再次校验
        if not repaired_poly.is_valid:
            if VISUALIZE_POLYGON_REPAIR:
                visualize_polygon_repair(source_filename, original_poly, repaired_poly, raw_points)
            logging.error(f"{source_filename} - 修复后仍无效，已跳过")
            return []

    # 若开启可视化，绘制修复前后对比（无论原始是否有效）
    # if VISUALIZE_POLYGON_REPAIR:
    #     visualize_polygon_repair(source_filename, original_poly, repaired_poly, raw_points)

    wall_polygon = repaired_poly

    # 3. 计算每个bbox的结果（核心逻辑不变，新增source_file字段）
    bbox_results = []

    # 处理无bbox的情况
    if not bboxes:
        bbox_results.append({
            "source_file": source_filename,
            "bbox_id": "None",
            "class_name": "None",
            "fully_inside_no_touch": "False",
            "inside_or_on_boundary": "False",
            "practically_inside": "False",
            "overlap_ratio": 0.0,
            "total_area": 0.0,
            "outside_area": 0.0,
            "intersection_ratio": 0.0
        })
        return bbox_results

    for bbox in bboxes:
        bbox_poly = get_bbox_polygon(bbox)
        overlap_ratio, _, outside_part, outside_area_raw = calculate_overlap_ratio(bbox_poly, wall_polygon)
        total_area = bbox_poly.area

        # 判定逻辑（不变）
        fully_inside_no_touch = wall_polygon.contains(bbox_poly)
        inside_or_on_boundary = wall_polygon.covers(bbox_poly)
        has_ignorable_outside = (outside_area_raw > 0) and (outside_area_raw < MIN_OUTSIDE_AREA)
        practically_inside = inside_or_on_boundary or has_ignorable_outside
        intersection = bbox_poly.intersection(wall_polygon)
        intersection_ratio = min(intersection.area / total_area if total_area > 0 else 0.0, 1.0)

        # 组装结果：新增source_file字段，标记数据来自哪个txt
        bbox_results.append({
            "source_file": source_filename,  # 关键新增：区分不同文件的结果
            "bbox_id": bbox.id,
            "class_name": bbox.class_name,
            "fully_inside_no_touch": "True" if fully_inside_no_touch else "False",
            "inside_or_on_boundary": "True" if inside_or_on_boundary else "False",
            "practically_inside": "True" if practically_inside else "False",
            "overlap_ratio": round(max(0.0, overlap_ratio), 9),
            "total_area": round(total_area, 9),
            "outside_area": round(outside_area_raw, 9),
            "intersection_ratio": round(intersection_ratio, 9)
        })
    
    # 4. 可视化bbox与墙体的重叠关系
    if VISUALIZE_WALL_BBOX_OVERLAP:
        visualize_wall_bbox_overlap(walls, bboxes, wall_polygon, title_suffix=f"({source_filename})")
    
    print(f"已处理：{source_filename}（共{len(bbox_results)}个bbox）")
    return bbox_results


def main(input_parent_dir, output_csv_path):
    """
    处理父文件夹下所有子文件夹中的txt文件，汇总结果到单个CSV
    :param input_parent_dir: 存放子文件夹的父文件夹路径
    :param output_csv_path: 最终汇总的CSV文件路径（含文件名，如"results/all_summary.csv"）
    """
    # 1. 递归遍历父文件夹下所有子文件夹
    all_bbox_results = []
    txt_files_cnt = 0
    for root, dirs, files in os.walk(input_parent_dir):
        txt_files = [f for f in files if f.lower().endswith(".txt")]
        
        if not txt_files:
            continue  # 如果当前子文件夹中没有txt文件，跳过

        print(f"正在处理文件夹：{root}，找到 {len(txt_files)} 个 .txt 文件")

        # 2. 遍历当前子文件夹中的txt文件，调用process_file计算并汇总结果
        for txt_filename in txt_files:
            txt_file_path = os.path.join(root, txt_filename)
            file_results = process_file(txt_file_path)
            if file_results:  # 只添加有效结果
                all_bbox_results.extend(file_results)
            txt_files_cnt += 1
    
    # 3. 写入汇总CSV（确保输出目录存在）
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # 自动创建不存在的输出目录
    
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        # 字段名需包含新增的"source_file"
        fieldnames = [
            "source_file", "bbox_id", "class_name", "fully_inside_no_touch",
            "inside_or_on_boundary", "practically_inside", "overlap_ratio",
            "total_area", "outside_area", "intersection_ratio"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_bbox_results)  # 批量写入所有结果
    
    # 4. 输出处理总结
    print(f"\n===== 批量处理完成 =====")
    print(f"处理父文件夹：{input_parent_dir}")
    print(f"处理txt文件数：{txt_files_cnt}")  # 这里可能需要重新计算文件数
    print(f"汇总bbox结果数：{len(all_bbox_results)}")
    print(f"汇总CSV保存路径：{output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python script.py <input_parent_folder> <output_summary_csv>")
        print("示例：[处理单个chunk的文件夹] python script.py ./chunk000_txt ./results/chunk000_summary.csv")
        print("示例：[递归处理所有文件夹]    python script.py ./layout ./results/all_bbox_summary.csv")
        print("说明：input_parent_folder 为存放子文件夹的父文件夹；output_summary_csv 为最终汇总的CSV路径（含文件名）")
        exit(1)
    
    input_parent_dir = sys.argv[1]
    output_csv_path = sys.argv[2]
    
    if not os.path.isdir(input_parent_dir):
        print(f"错误：{input_parent_dir} 不是有效的文件夹，请检查路径")
        exit(1)
    
    main(input_parent_dir, output_csv_path)