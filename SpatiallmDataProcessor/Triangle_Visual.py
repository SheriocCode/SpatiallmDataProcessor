"""
Triangle_Visual.py
7张子图展示：
    1. 原始点云密度图
    2. 房间边界与边界框
    3. 延伸至边界的墙面线段
    4. 约束Delaunay三角剖分结果
    5. 按房间着色的三角网格
    6. COCO标注叠加密度图（带房间分组颜色
    7. COCO标注+三角剖分叠加显示

- 固定随机种子(seed=42)确保颜色一致性
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle
import json
import os
import cv2
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import random

# 复用原有功能函数
def save_triangulation_to_ply(vertices, triangles, filename='output_mesh.ply', binary=True):
    if vertices.ndim == 2 and vertices.shape[1] == 2:
        z = np.zeros((vertices.shape[0], 1), dtype=vertices.dtype)
        vertices = np.hstack([vertices, z])
    elif vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices 必须是 N×2 或 N×3 的数组，但得到了 {vertices.shape}")

    num_vertices = vertices.shape[0]
    num_faces = triangles.shape[0]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        "end_header"
    ]

    header = '\n'.join(header_lines) + '\n'

    with open(filename, 'wb' if binary else 'w') as f:
        if binary:
            f.write(header.encode('ascii'))
        else:
            f.write(header)

        if binary:
            vertices_flat = vertices.astype(np.float32).flatten()
            f.write(vertices_flat.tobytes())
        else:
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if binary:
            for face in triangles:
                face_header = np.array([3], dtype=np.uint8)
                face_indices = face.astype(np.int32)
                f.write(face_header.tobytes())
                f.write(face_indices.tobytes())
        else:
            for face in triangles:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"[INFO] 三角网格已保存为 PLY 文件: {os.path.abspath(filename)}")


def line_intersection(seg1, seg2):
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 平行线，无交点

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / denom
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    u = u_num / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def extend_segment_to_bbox(segment, bbox):
    (x1, y1), (x2, y2) = segment
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[2]

    bbox_edges = [
        [(min_x, min_y), (max_x, min_y)],  # 底边
        [(max_x, min_y), (max_x, max_y)],  # 右边
        [(max_x, max_y), (min_x, max_y)],  # 顶边
        [(min_x, max_y), (min_x, min_y)]   # 左边
    ]

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:
        x = x1
        return [(x, min_y), (x, max_y)]
    
    if dy == 0:
        y = y1
        return [(min_x, y), (max_x, y)]

    intersections = []
    for edge in bbox_edges:
        intersect = line_intersection(segment, edge)
        if intersect:
            intersections.append(intersect)

    if len(intersections) < 2:
        m = dy / dx
        b = y1 - m * x1

        y_left = m * min_x + b
        y_right = m * max_x + b

        x_bottom = (min_y - b) / m if m != 0 else None
        x_top = (max_y - b) / m if m != 0 else None

        if min_y <= y_left <= max_y:
            intersections.append((min_x, y_left))
        if min_y <= y_right <= max_y:
            intersections.append((max_x, y_right))
        if x_bottom is not None and min_x <= x_bottom <= max_x:
            intersections.append((x_bottom, min_y))
        if x_top is not None and min_x <= x_top <= max_x:
            intersections.append((x_top, max_y))

    if len(intersections) >= 2:
        dists = [((x - x1)**2 + (y - y1)** 2) for x, y in intersections]
        far1_idx = np.argmax(dists)
        far1 = intersections.pop(far1_idx)
        
        dists = [((x - far1[0])**2 + (y - far1[1])** 2) for x, y in intersections]
        far2_idx = np.argmax(dists)
        far2 = intersections[far2_idx]
        
        return [far1, far2]
    
    return segment

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)):
            x_intersect = (y - yi) * (xj - xi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
    return inside

def assign_triangles_to_rooms(triangles, vertices, room_polygons):
    num_triangles = len(triangles)
    triangle_room_ids = [-1] * num_triangles
    
    for tri_idx, tri in enumerate(triangles):
        p1 = vertices[tri[0]]
        p2 = vertices[tri[1]]
        p3 = vertices[tri[2]]
        centroid = np.mean([p1, p2, p3], axis=0)
        
        for room_id, polygon in room_polygons.items():
            if point_in_polygon(centroid, polygon):
                triangle_room_ids[tri_idx] = room_id
                break
    
    return triangle_room_ids

def calculate_area(poly):
    if len(poly) < 3:
        return 0
    return cv2.contourArea(poly.astype(np.int32))

# 存储每个子图的绘制函数和数据，用于放大显示
subplot_data = []

def draw_subplot_1(ax, density_image):
    flipped_image = np.flipud(density_image)
    ax.imshow(flipped_image)
    ax.set_title('1. 点云密度图', fontsize=12)
    ax.axis('off')

def draw_subplot_2(ax, box, annotations_filtered, room_color_map):
    box_x, box_y = zip(*box)
    ax.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    for ann in annotations_filtered:
        rid = ann["room_id"]
        seg = ann["segmentation"][0]
        polygon = [(seg[i*2], seg[i*2+1]) for i in range(len(seg)//2)]
        x, y = zip(*polygon)
        x += (x[0],)
        y += (y[0],)
        ax.plot(x, y, '-', color=room_color_map[rid], linewidth=1.5,
                label=f'房间 {rid}' if rid not in [l.get_label() for l in ax.get_legend_handles_labels()[0]] else "")
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.set_title('2. 原始房间边界', fontsize=12)

def draw_subplot_3(ax, box, wall_segments):
    box_x, box_y = zip(*box)
    ax.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    for seg in wall_segments:
        x, y = zip(*seg)
        ax.plot(x, y, 'g-', linewidth=1.5, label='延伸后墙面线段' if ax.get_legend_handles_labels()[1] == [] else "")
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.set_title('3. 延伸至边界的墙面线段', fontsize=12)

def draw_subplot_4(ax, box, triang, vertices):
    ax.triplot(triang, color='lightblue', linewidth=0.5)
    ax.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)
    box_x, box_y = zip(*box)
    ax.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.set_title('4. 约束Delaunay三角剖分结果', fontsize=12)

def draw_subplot_5(ax, box, triang, adjusted_ids, cmap, wall_segments, room_polygons, cmap_colors):
    ax.tripcolor(triang, adjusted_ids, cmap=cmap, alpha=0.7)
    ax.triplot(triang, color='k', linewidth=0.5)
    box_x, box_y = zip(*box)
    ax.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    for seg in wall_segments:
        x, y = zip(*seg)
        ax.plot(x, y, 'g-', linewidth=1.5, label='墙面线段' if ax.get_legend_handles_labels()[1] == [] else "")
    handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in cmap_colors[:-1]]
    labels = [f'房间 {rid}' for rid in sorted(room_polygons.keys())]
    handles.append(plt.Rectangle((0,0),1,1, facecolor=cmap_colors[-1]))
    labels.append('墙体/外部区域')
    ax.legend(handles, labels, loc='best', fontsize=8)
    ax.set_aspect('equal')
    ax.set_title('5. 按房间着色的三角剖分结果', fontsize=12)

def draw_subplot_6(ax, density_image, annotations, categories, image_height):
    """
    在点云密度图上绘制COCO格式标注（子图6）
    
    该函数将房间、墙体、门窗等标注以不同样式绘制在密度图上：
    - 房间：半透明填充多边形 + 虚线边界框
    - 门窗：加粗线段 + 白色端点标记
    - 所有标注均包含白色背景标签
    
    参数:
        ax (matplotlib.axes.Axes): 目标绘图轴对象
        density_image (numpy.ndarray): 点云密度图数据 (H×W×3)
        annotations (list): COCO标注列表，每个元素为包含segmentation、bbox等信息的字典
        categories (list): 类别定义列表，元素为{'id': int, 'name': str}格式
        image_height (int): 图像高度，用于Y坐标翻转校正
    
    功能特性:
        - 使用随机种子42确保颜色一致性
        - 按room_id分组，同一房间使用相近色系
        - 自动识别线段标注(area=0或坐标数<6)
        - 动态生成带类别名的图例
    """
    random.seed(42)
    
    # 翻转图像
    flipped_image = np.flipud(density_image)
    ax.imshow(flipped_image)
    
    # 创建基础颜色映射（确定性）
    category_ids = sorted(set(ann['category_id'] for ann in annotations))
    np.random.seed(42)
    base_color_map = {cat_id: np.random.rand(3,) for cat_id in category_ids}
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # 按room_id分组标注
    annotations_by_room = {}
    for ann in annotations:
        room_id = ann.get('room_id', 0)
        if room_id not in annotations_by_room:
            annotations_by_room[room_id] = []
        annotations_by_room[room_id].append(ann)
    
    # 收集图例项
    legend_items = {}
    
    # 绘制每个房间的标注
    for room_id, anns in annotations_by_room.items():
        # 为每个房间生成颜色偏移（确定性）
        room_color_shift = random.uniform(0.4, 1.0)
        
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = cat_id_to_name.get(cat_id, f'类别 {cat_id}' if cat_id not in [0, 1] else f'类别{cat_id}(Door/Window)')
            base_color = base_color_map[cat_id]
            final_color = tuple(min(1.0, c * room_color_shift) for c in base_color)
            
            # 判断是否为线段标注
            is_line = (ann.get('area', 0) == 0 or 
                      len(ann.get('segmentation', [[]])[0]) < 6 or
                      ann.get('bbox', [0,0,0,0])[3] == 0)
            
            # 绘制线段（door/window）
            if is_line:
                for seg in ann['segmentation']:
                    if len(seg) < 4:
                        continue
                    points = np.array(seg).reshape(-1, 2)
                    points[:, 1] = image_height - points[:, 1]  # 翻转y坐标
                    
                    if len(points) == 2:  # 线段
                        ax.plot(points[:, 0], points[:, 1], 
                                color=final_color, 
                                linewidth=3,
                                marker='o', markersize=5,
                                markerfacecolor='white',
                                markeredgecolor=final_color,
                                markeredgewidth=1.5)
                        
                        # 中点标签
                        mid = points.mean(axis=0)
                        label = f"Room{room_id}-{cat_name}"
                        ax.text(mid[0], mid[1] - 8, label, 
                                color='white', fontsize=7, ha='center',
                                bbox=dict(boxstyle='round', facecolor=final_color, 
                                         alpha=0.85, edgecolor='white', linewidth=0.5))
            
            # 绘制多边形（房间）
            else:
                for seg in ann['segmentation']:
                    if len(seg) < 6:
                        continue
                    poly = np.array(seg).reshape(-1, 2)
                    poly[:, 1] = image_height - poly[:, 1]  # 翻转y坐标
                    
                    if len(poly) < 3 or cv2.contourArea(poly.astype(np.int32)) < 1.0:
                        continue
                    
                    # 填充多边形
                    patch = Polygon(poly, closed=True, 
                                   facecolor=final_color, 
                                   alpha=0.4, 
                                   edgecolor='white', 
                                   linewidth=1)
                    ax.add_patch(patch)
                
                # 绘制边界框
                if 'bbox' in ann and len(ann['bbox']) == 4:
                    x, y, w, h = ann['bbox']
                    y = image_height - y - h
                    rect = plt.Rectangle((x, y), w, h, 
                                       linewidth=2,
                                       edgecolor=final_color, 
                                       facecolor='none',
                                       linestyle='--')
                    ax.add_patch(rect)
                    
                    # 标签
                    label = f"Room{room_id}-{cat_name}"
                    ax.text(x, y - 5, label, 
                            color='white', fontsize=7, ha='left',
                            bbox=dict(boxstyle='round', facecolor=final_color, 
                                     alpha=0.85, edgecolor='white', linewidth=0.5))
            
            # 记录图例
            if cat_name not in legend_items:
                legend_items[cat_name] = plt.Rectangle((0,0),1,1, facecolor=base_color, alpha=0.4)
    
    # 添加图例
    if legend_items:
        ax.legend(legend_items.values(), legend_items.keys(), 
                 loc='best', fontsize=8, framealpha=0.9)
    
    ax.set_title('6. 点云密度图与标注', fontsize=12)
    ax.axis('off')


def draw_subplot_7(ax, density_image, triang, adjusted_ids, cmap, annotations, categories, image_height):
    """
    在点云密度图上叠加三角剖分结果和COCO标注（子图7）
    
    与draw_subplot_6类似，但额外显示：
    - 底层三角网格 (30%透明度)
    - 房间区域按三角剖分ID着色
    - 标注覆盖在三角网格之上
    
    参数:
        ax (matplotlib.axes.Axes): 目标绘图轴对象
        density_image (numpy.ndarray): 点云密度图数据 (H×W×3)
        triang (matplotlib.tri.Triangulation): 三角剖分对象
        adjusted_ids (list): 三角形所属房间ID列表
        cmap (matplotlib.colors.ListedColormap): 房间颜色映射表
        annotations (list): COCO标注列表
        categories (list): 类别定义列表
        image_height (int): 图像高度，用于Y坐标翻转
    
    坐标处理:
        - 三角网格顶点Y坐标会翻转匹配图像坐标系
        - 标注坐标同步翻转保持对齐
    """
    random.seed(42)
    
    # 翻转图像
    flipped_image = np.flipud(density_image)
    ax.imshow(flipped_image)
    
    # 三角剖分坐标翻转
    tri_vertices = np.array([triang.x, triang.y]).T
    tri_vertices[:, 1] = image_height - tri_vertices[:, 1]
    flipped_triang = mtri.Triangulation(
        tri_vertices[:, 0], 
        tri_vertices[:, 1], 
        triang.triangles
    )
    ax.tripcolor(flipped_triang, adjusted_ids, cmap=cmap, alpha=0.3)
    ax.triplot(flipped_triang, color='k', linewidth=0.3, alpha=0.5)
    
    # 创建基础颜色映射（确定性）
    category_ids = sorted(set(ann['category_id'] for ann in annotations))
    np.random.seed(42)
    base_color_map = {cat_id: np.random.rand(3,) for cat_id in category_ids}
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # 按room_id分组标注
    annotations_by_room = {}
    for ann in annotations:
        room_id = ann.get('room_id', 0)
        if room_id not in annotations_by_room:
            annotations_by_room[room_id] = []
        annotations_by_room[room_id].append(ann)
    
    # 收集图例项
    legend_items = {}
    
    # 绘制每个房间的标注
    for room_id, anns in annotations_by_room.items():
        # 为每个房间生成颜色偏移（确定性）
        room_color_shift = random.uniform(0.4, 1.0)
        
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = cat_id_to_name.get(cat_id, f'类别 {cat_id}')
            base_color = base_color_map[cat_id]
            final_color = tuple(min(1.0, c * room_color_shift) for c in base_color)
            
            # 判断是否为线段标注
            is_line = (ann.get('area', 0) == 0 or 
                      len(ann.get('segmentation', [[]])[0]) < 6 or
                      ann.get('bbox', [0,0,0,0])[3] == 0)
            
            # 绘制线段（door/window）
            if is_line:
                for seg in ann['segmentation']:
                    if len(seg) < 4:
                        continue
                    points = np.array(seg).reshape(-1, 2)
                    points[:, 1] = image_height - points[:, 1]
                    
                    if len(points) == 2:  # 线段
                        ax.plot(points[:, 0], points[:, 1], 
                                color=final_color, 
                                linewidth=3,
                                marker='o', markersize=5,
                                markerfacecolor='white',
                                markeredgecolor=final_color,
                                markeredgewidth=1.5)
            
            # 绘制多边形（房间）
            else:
                for seg in ann['segmentation']:
                    if len(seg) < 6:
                        continue
                    poly = np.array(seg).reshape(-1, 2)
                    poly[:, 1] = image_height - poly[:, 1]
                    
                    if len(poly) < 3 or cv2.contourArea(poly.astype(np.int32)) < 1.0:
                        continue
                    
                    # 填充多边形
                    patch = Polygon(poly, closed=True, 
                                   facecolor=final_color, 
                                   alpha=0.5, 
                                   edgecolor='white', 
                                   linewidth=1.5)
                    ax.add_patch(patch)
                
                # 绘制边界框
                if 'bbox' in ann and len(ann['bbox']) == 4:
                    x, y, w, h = ann['bbox']
                    y = image_height - y - h
                    rect = plt.Rectangle((x, y), w, h, 
                                       linewidth=2,
                                       edgecolor=final_color, 
                                       facecolor='none',
                                       linestyle='--')
                    ax.add_patch(rect)
            
            # 记录图例
            if cat_name not in legend_items:
                legend_items[cat_name] = plt.Rectangle((0,0),1,1, facecolor=base_color, alpha=0.4)
    
    # 添加图例
    if legend_items:
        ax.legend(legend_items.values(), legend_items.keys(), 
                 loc='best', fontsize=8, framealpha=0.9)
    
    ax.set_title('7. 点云密度图+标注+三角剖分', fontsize=12)
    ax.axis('off')

# 点击事件处理函数 - 放大显示子图
def on_click(event):
    if event.inaxes is None:  # 点击不在任何子图上
        return
    
    # 找到被点击的子图索引
    for i, ax in enumerate(fig.axes):
        if ax == event.inaxes:
            # 创建新窗口显示放大后的子图
            zoom_fig, zoom_ax = plt.subplots(figsize=(10, 10))
            # 调用对应的绘制函数
            subplot_data[i](zoom_ax)
            zoom_fig.tight_layout()
            zoom_fig.canvas.manager.set_window_title(f'放大视图: {zoom_ax.get_title()}')
            plt.show()
            break

def visualize_combined_results(json_path, density_map_path, output_ply="combined_mesh.ply"):
    global fig, subplot_data  # 全局变量用于事件处理
    
    # 1. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 提取COCO数据中的关键部分
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])  # 获取类别信息
    annotations_filtered = [ann for ann in annotations if 'room_id' in ann and ann.get('category_id') not in [0, 1]]
    
    # 读取点云密度图并获取图像高度（用于坐标转换）
    density_image = cv2.imread(density_map_path, cv2.IMREAD_COLOR)
    if density_image is None:
        raise ValueError(f"无法读取密度图: {density_map_path}")
    density_image = cv2.cvtColor(density_image, cv2.COLOR_BGR2RGB)
    image_height = density_image.shape[0]  # 获取图像高度，用于后续坐标转换
    
    # 定义外包围盒
    box = [(0, 0), (256, 0), (256, 256), (0, 256)]
    min_x, min_y = box[0]
    max_x, max_y = box[2]
    
    # 提取所有墙面线段并延伸至边界框
    wall_segments = []
    original_segments = []
    all_points = []
    room_polygons = {}
    
    for ann in annotations_filtered:
        room_id = ann["room_id"]
        seg = ann["segmentation"][0]
        polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
        
        if room_id not in room_polygons:
            room_polygons[room_id] = polygon
        else:
            room_polygons[room_id].extend(polygon)
        
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            original_segments.append((p1, p2))
            extended = extend_segment_to_bbox((p1, p2), box)
            wall_segments.append(extended)
            all_points.extend(extended)
    
    # 处理顶点和线段
    all_points.extend(box)
    unique_points = []
    seen = set()
    for p in all_points:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # 去重线段
    unique_segments = []
    seen_seg = set()
    for seg in wall_segments:
        p1, p2 = seg
        key = tuple(sorted([
            (round(p1[0], 6), round(p1[1], 6)),
            (round(p2[0], 6), round(p2[1], 6))
        ]))
        if key not in seen_seg:
            seen_seg.add(key)
            unique_segments.append(seg)
    wall_segments = unique_segments
    
    # 构建线段索引
    point_indices = { (round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(unique_points) }
    segments = []
    for seg in wall_segments:
        p1, p2 = seg
        idx1 = point_indices[(round(p1[0], 6), round(p1[1], 6))]
        idx2 = point_indices[(round(p2[0], 6), round(p2[1], 6))]
        segments.append((idx1, idx2))
    
    # 添加边界框的边
    box_indices = [point_indices[(round(p[0], 6), round(p[1], 6))] for p in box]
    n = len(box_indices)
    for i in range(n):
        segments.append((box_indices[i], box_indices[(i + 1) % n]))
    
    # 执行三角剖分
    A = dict(
        vertices=np.array(unique_points, dtype=float),
        segments=np.array(segments, dtype=int)
    )
    B = triangle.triangulate(A, 'p')
    
    if 'triangles' not in B:
        raise ValueError("三角剖分失败，未生成任何三角形")
    
    vertices = B['vertices']
    triangles = B['triangles']
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    triangle_room_ids = assign_triangles_to_rooms(triangles, vertices, room_polygons)
    
    # 保存PLY文件
    save_triangulation_to_ply(vertices, triangles, filename=output_ply, binary=True)
    
    # 创建颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(room_polygons)))
    room_color_map = {rid: colors[i] for i, rid in enumerate(room_polygons.keys())}
    cmap_colors = [room_color_map[rid] for rid in sorted(room_polygons.keys())]
    cmap_colors.append([0.8, 0.8, 0.8])  # 灰色用于未分配区域
    cmap = ListedColormap(cmap_colors)
    adjusted_ids = [rid if rid != -1 else len(room_polygons) for rid in triangle_room_ids]
    
    # 存储每个子图的绘制函数（带参数）
    subplot_data = [
        lambda ax: draw_subplot_1(ax, density_image),
        lambda ax: draw_subplot_2(ax, box, annotations_filtered, room_color_map),
        lambda ax: draw_subplot_3(ax, box, wall_segments),
        lambda ax: draw_subplot_4(ax, box, triang, vertices),
        lambda ax: draw_subplot_5(ax, box, triang, adjusted_ids, cmap, wall_segments, room_polygons, cmap_colors),
        lambda ax: draw_subplot_6(ax, density_image, annotations, categories, image_height),
        lambda ax: draw_subplot_7(ax, density_image, triang, adjusted_ids, cmap, annotations, categories, image_height)
    ]
    
    # 创建7个子图
    fig = plt.figure(figsize=(24, 18))
    
    # 绘制子图
    ax1 = fig.add_subplot(331)
    subplot_data[0](ax1)
    
    ax2 = fig.add_subplot(332)
    subplot_data[1](ax2)
    
    ax3 = fig.add_subplot(333)
    subplot_data[2](ax3)
    
    ax4 = fig.add_subplot(334)
    subplot_data[3](ax4)
    
    ax5 = fig.add_subplot(335)
    subplot_data[4](ax5)
    
    ax6 = fig.add_subplot(336)
    subplot_data[5](ax6)
    
    ax7 = fig.add_subplot(337)
    subplot_data[6](ax7)
    
    # 绑定点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.savefig('combined_visualization.png', dpi=300, bbox_inches='tight')
    print("所有可视化结果已保存为 combined_visualization.png")
    print("提示：点击任何子图可查看放大版本")
    plt.show()

# 使用示例
if __name__ == "__main__":
    scene_name = 'scene_000005'  # 替换为你的场景名称
    JSON_PATH = f"coco_with_scaled/sample0_256/anno/{scene_name}.json"
    DENSITY_MAP_PATH = f"coco_with_scaled/sample0_256/density_map/{scene_name}.png"
    
    visualize_combined_results(
        json_path=JSON_PATH,
        density_map_path=DENSITY_MAP_PATH,
        output_ply="combined_mesh.ply"
    )