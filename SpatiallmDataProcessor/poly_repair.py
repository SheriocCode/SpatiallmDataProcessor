#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Contact       : ZhangJie
# @ Date         : 2025-11-05
# @ Version      : 0.0.1
"""
====================================================================
Wall Polygon Repair and Validation Tool
====================================================================
介绍：
解析场景txt文件中的墙体数据，检测并修复墙体构成的多边形有效性
问题（共线顶点、自相交、未闭合等），输出修复后的墙体数据并生成
详细处理日志与可视化对比图。

输入：
- 单个scene_*.txt文件路径，或
- 包含多个scene_*.txt文件的文件夹路径

输出：
1. 修复后的txt文件
2. 处理日志
3. 多边形修复对比图和pickle文件
"""

import sys
import os
import pickle
from pathlib import Path
import logging
import argparse
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from utils.entity import Wall, Bbox
from utils.dir_util import make_dirs
from utils.log_util import init_logger


def parse_scene_file(filename):
    """解析txt文件，提取墙体和边界框数据"""
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
    """从墙体数据创建多边形"""
    if not walls:
        return None
        
    polygon_points = [(wall.ax, wall.ay) for wall in walls]
    
    # 确保多边形闭合
    if polygon_points and polygon_points[0] != polygon_points[-1]:
        polygon_points.append(polygon_points[0])
        
    return Polygon(polygon_points)


def analyze_polygon_issues(source_filename, wall_polygon, walls):
    """分析多边形无效原因并记录日志"""
    if wall_polygon is None:
        error_msg = f"{source_filename} - 无墙体数据，无法创建多边形"
        logging.warning(error_msg)
        return False

    if wall_polygon.is_valid:
        return True

    # 提取原始顶点信息
    raw_points = [(wall.ax, wall.ay) for wall in walls] if walls else []
    if raw_points and raw_points[0] != raw_points[-1]:
        raw_points.append(raw_points[0])
    point_count = len(raw_points)

    invalid_reason = explain_validity(wall_polygon)
    error_details = []

    # 顶点数量检查
    if point_count < 3:
        error_details.append(f"顶点数量不足（{point_count}个，至少需要3个）")

    # 连续共线检查
    def has_continuous_collinear(points):
        if len(points) < 3:
            return False
        for i in range(len(points) - 2):
            p1, p2, p3 = points[i], points[i+1], points[i+2]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            if abs(cross) < 1e-9:
                return True
        return False
    if has_continuous_collinear(raw_points):
        error_details.append("存在连续共线顶点")

    # 自相交检查
    if "Self-intersection" in invalid_reason:
        pos = invalid_reason.split("at ")[-1].strip() if "at " in invalid_reason else "未知位置"
        error_details.append(f"自相交（位置：{pos}）")

    # 闭合性检查
    if not wall_polygon.exterior.is_closed:
        error_details.append("多边形未闭合")

    if not error_details:
        error_details.append(f"其他错误：{invalid_reason}")

    full_msg = f"{source_filename} - 多边形无效（顶点数：{point_count}）：{'; '.join(error_details)}"
    logging.warning(full_msg)
    return False


def repair_invalid_polygon(source_filename, wall_polygon, walls):
    """尝试修复无效多边形"""
    if wall_polygon is None:
        logging.warning(f"{source_filename} - 无多边形可修复")
        return None

    # 提取原始顶点
    raw_points = [(wall.ax, wall.ay) for wall in walls] if walls else []
    if raw_points and raw_points[0] != raw_points[-1]:
        raw_points.append(raw_points[0])

    # 顶点数量不足处理
    if len(raw_points) < 3:
        logging.warning(f"{source_filename} - 顶点不足，无法修复")
        return None

    # 移除连续共线顶点
    def remove_collinear_points(points):
        if len(points) <= 3:
            return points
        filtered = [points[0]]
        for i in range(1, len(points)-1):
            p1, p2, p3 = filtered[-1], points[i], points[i+1]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            if abs(cross) >= 1e-9:
                filtered.append(p2)
        filtered.append(points[-1])
        if filtered[0] != filtered[-1]:
            filtered.append(filtered[0])
        return filtered

    # 检查并处理共线问题
    has_collinear = any(
        abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])) < 1e-9
        for i, (p1, p2, p3) in enumerate(zip(raw_points[:-2], raw_points[1:-1], raw_points[2:]))
    )
    if has_collinear:
        repaired_points = remove_collinear_points(raw_points)
        wall_polygon = Polygon(repaired_points)
        logging.info(f"{source_filename} - 移除共线顶点（{len(raw_points)}→{len(repaired_points)}）")
        if wall_polygon.is_valid:
            return wall_polygon

    # 处理自相交问题
    invalid_reason = explain_validity(wall_polygon)
    if "Self-intersection" in invalid_reason:
        try:
            repaired_poly = wall_polygon.buffer(0)
            # 处理多多边形情况
            if repaired_poly.geom_type == "MultiPolygon":
                largest_poly = max(repaired_poly.geoms, key=lambda g: g.area)
                repaired_poly = largest_poly
                logging.info(f"{source_filename} - 选择最大子多边形（面积：{largest_poly.area:.4f}）")
            
            if repaired_poly.is_valid and repaired_poly.geom_type == "Polygon":
                logging.info(f"{source_filename} - 修复自相交问题")
                return repaired_poly
        except Exception as e:
            logging.error(f"{source_filename} - 自相交修复失败：{str(e)}")

    # 处理未闭合问题
    if not wall_polygon.exterior.is_closed:
        closed_points = list(wall_polygon.exterior.coords)
        if closed_points and closed_points[0] != closed_points[-1]:
            closed_points.append(closed_points[0])
            repaired_poly = Polygon(closed_points)
            if repaired_poly.is_valid:
                logging.info(f"{source_filename} - 修复未闭合问题")
                return repaired_poly

    logging.error(f"{source_filename} - 所有修复尝试失败")
    return None


def update_walls_from_polygon(original_walls, repaired_poly):
    """根据修复后的多边形更新墙体数据"""
    if not repaired_poly or not original_walls:
        return []

    # 提取修复后的顶点（去除最后一个闭合点）
    coords = list(repaired_poly.exterior.coords[:-1])
    if len(coords) < 2:
        return []

    # 创建新的墙体列表（保持原始ID顺序和非坐标属性）
    new_walls = []
    for i in range(len(coords)):
        # 循环连接顶点（最后一个顶点连接到第一个）
        next_i = (i + 1) % len(coords)
        ax, ay = coords[i]
        bx, by = coords[next_i]

        # 保留原始墙体的非坐标属性
        original_wall = original_walls[i % len(original_walls)]
        new_wall = Wall(
            id=i,
            ax=ax,
            ay=ay,
            az=original_wall.az,
            bx=bx,
            by=by,
            bz=original_wall.bz,
            height=original_wall.height,
            thickness=original_wall.thickness
        )
        new_walls.append(new_wall)

    return new_walls


def write_repaired_file(original_path, walls, bboxes):
    """写入修复后的txt文件"""
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    # repaired_name = f"{name}_repaired{ext}"
    repaired_name = f"{name}{ext}"
    output_path = os.path.join(REPAIRED_TXT_PATH, repaired_name)

    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入墙体数据
        for wall in walls:
            wall_str = (f"wall_{wall.id}=Wall({wall.ax}, {wall.ay}, {wall.az}, "
                       f"{wall.bx}, {wall.by}, {wall.bz}, {wall.height}, {wall.thickness})")
            f.write(wall_str + '\n')
        
        # 写入边界框数据（保持不变）
        for bbox in bboxes:
            bbox_str = (f"bbox_{bbox.id}=Bbox({bbox.class_name}, {bbox.position_x}, {bbox.position_y}, "
                       f"{bbox.position_z}, {bbox.angle_z}, {bbox.scale_x}, {bbox.scale_y}, {bbox.scale_z})")
            f.write(bbox_str + '\n')

    return output_path


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

    pure_name = Path(source_filename).stem

    # 1. 保存修复前后对比图片
    if POLYGON_REPAIR_SAVE_FIG:
        out_path  = FIG_SAVE_PATH / f"{pure_name}_repair.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')

    # 2. 保存pickle文件
    if POLYGON_REPAIR_SAVE_PICKLE:
        pickle_path = PICKLE_SAVE_PATH / f"{pure_name}_repair.fig.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)

    plt.close(fig)


def load_and_visualize_polygon_repair(pickle_path, show=True):
    """
    从pickle文件加载并显示可视化结果
    :param pickle_path: pickle文件路径
    :param show: 是否显示图形（默认为True）
    """
    pickle_path = Path(pickle_path)
    
    if not pickle_path.exists():
        print(f"错误: 文件不存在 {pickle_path}")
        return None

    try:
        with open(pickle_path, 'rb') as f:
            fig = pickle.load(f)
        
        if show:
            plt.show()
        
        return fig
    except Exception as e:
        print(f"加载失败: {str(e)}")
        return None


def process_single_file(file_path):
    """处理单个文件的主函数"""
    try:
        # logging.info(f"开始处理文件：{file_path}")
        source_filename = os.path.basename(file_path)
        walls, bboxes = parse_scene_file(file_path)
        # 提取原始顶点（用于可视化）
        raw_points = [(wall.ax, wall.ay) for wall in walls] if walls else []

        # 创建原始多边形并检查有效性
        original_poly = create_wall_polygon(walls)
        is_valid = analyze_polygon_issues(os.path.basename(file_path), original_poly, walls)

        if is_valid:
            return None

        # 修复多边形（如果无效）
        repaired_poly = original_poly if is_valid else repair_invalid_polygon(
            os.path.basename(file_path), original_poly, walls)

        if not repaired_poly or not repaired_poly.is_valid:
            logging.error(f"{file_path} - 无法修复为有效多边形，跳过输出")
            return None

        # 根据修复后的多边形更新墙体数据
        new_walls = update_walls_from_polygon(walls, repaired_poly)
        if not new_walls:
            logging.error(f"{file_path} - 无法生成有效墙体数据，跳过输出")
            return None

        # 写入修复后的文件
        output_path = write_repaired_file(file_path, new_walls, bboxes)
        logging.info(f"{file_path} - 修复完成，输出至：{output_path}")
        # 写入csv文件


        visualize_polygon_repair(source_filename, original_poly, repaired_poly, raw_points)

        return output_path

    except Exception as e:
        logging.error(f"{file_path} - 处理失败：{str(e)}", exc_info=True)
        return None


def main(input_path):
    """主函数：处理单个文件或文件夹"""
    if os.path.isfile(input_path) and input_path.endswith('.txt'):
        # 处理单个文件
        process_single_file(input_path)
    elif os.path.isdir(input_path):
        # 处理文件夹中的所有txt文件
        txt_files = [f for f in os.listdir(input_path) 
                    if os.path.isfile(os.path.join(input_path, f)) 
                    and f.endswith('.txt')]
        
        if not txt_files:
            logging.warning(f"文件夹 {input_path} 中未找到txt文件")
            print(f"警告：文件夹 {input_path} 中未找到txt文件")
            return

        for txt_file in txt_files:
            file_path = os.path.join(input_path, txt_file)
            process_single_file(file_path)
        
        print(f"处理完成，共处理 {len(txt_files)} 个文件")
        print(f"日志记录已保存至 .log")
    else:
        print(f"错误：{input_path} 不是有效的文件或文件夹")
        logging.error(f"无效输入路径：{input_path}")


if __name__ == "__main__":
    # ==================== 路径配置 ====================
    OUTPUT_PATH = Path("poly_repair_output")
    REPAIRED_TXT_PATH = OUTPUT_PATH / "txt_repaired" # 修复后的txt文件路径
    FIG_SAVE_PATH = OUTPUT_PATH / "fig_save"
    PICKLE_SAVE_PATH = OUTPUT_PATH / "pickle_save"
    make_dirs(OUTPUT_PATH, REPAIRED_TXT_PATH, FIG_SAVE_PATH, PICKLE_SAVE_PATH)

    # ==================== 日志配置 ====================
    init_logger(filename= OUTPUT_PATH / '.log')

    # ==================== 功能开关 (保存修复前后的对比图)====================
    POLYGON_REPAIR_SAVE_FIG = True      # 保存PNG图片
    POLYGON_REPAIR_SAVE_PICKLE = True   # 保存pickle文件，后续可以通过 --mode visualize 可视化

    parser = argparse.ArgumentParser(
        description="多边形修复工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    使用示例:
    # 1. 批量处理 / 处理单个文件
    python script.py --mode process <input_path>
    $ python script.py --mode process ./chunk000_txt
    $ python script.py --mode process ./scene_001.txt
    
    # 2. 可视化
    python script.py --mode visualize <pickle_path>
    $ python script.py --mode visualize poly_repair_output/pickle_save/scene_001_repair.fig.pkl
    """
    )

    parser.add_argument("--mode", type=str, default="process", choices=["process", "visualize"], 
                        help="运行模式（process/visualize）")
    parser.add_argument("input_path", type=str, help="输入路径（process模式下为文件/文件夹，visualize模式下为pickle文件）")

    args = parser.parse_args()

    if args.mode == "process":
        main(args.input_path)
    elif args.mode == "visualize":
        load_and_visualize_polygon_repair(args.input_path, show=True) 
