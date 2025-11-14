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
    """将三角剖分结果保存为 PLY 文件格式"""
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
    """计算两条线段的交点"""
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
    """将线段延伸至边界框"""
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
        # 计算每个交点到线段起点的距离
        dists = [((x - x1)**2 + (y - y1)** 2) for x, y in intersections]
        # 找到最远点的索引并移除
        far1_idx = np.argmax(dists)
        far1 = intersections.pop(far1_idx)
        
        # 计算剩余点到far1的距离
        dists = [((x - far1[0])**2 + (y - far1[1])** 2) for x, y in intersections]
        # 找到最远点的索引
        far2_idx = np.argmax(dists)
        far2 = intersections[far2_idx]
        
        return [far1, far2]
    
    return segment

def point_in_polygon(point, polygon):
    """判断点是否在多边形内部（射线法）"""
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
    """根据room_id将三角形分配到对应的房间"""
    num_triangles = len(triangles)
    triangle_room_ids = [-1] * num_triangles  # -1表示未分配（可能是墙体或外部区域）
    
    for tri_idx, tri in enumerate(triangles):
        # 计算三角形重心（用于判断所属房间）
        p1 = vertices[tri[0]]
        p2 = vertices[tri[1]]
        p3 = vertices[tri[2]]
        centroid = np.mean([p1, p2, p3], axis=0)
        
        # 检查重心属于哪个房间的多边形
        for room_id, polygon in room_polygons.items():
            if point_in_polygon(centroid, polygon):
                triangle_room_ids[tri_idx] = room_id
                break  # 一个三角形只属于一个房间
    
    return triangle_room_ids

def calculate_area(poly):
    """计算多边形面积"""
    if len(poly) < 3:
        return 0
    return cv2.contourArea(poly.astype(np.int32))

def visualize_combined_results(json_path, density_map_path, output_ply="combined_mesh.ply"):
    """整合所有可视化需求"""
    # 1. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    annotations = coco_data.get('annotations', [])
    annotations_filtered = [ann for ann in annotations if 'room_id' in ann and ann.get('category_id') not in [0, 1]]
    
    # 读取点云密度图
    density_image = cv2.imread(density_map_path, cv2.IMREAD_COLOR)
    if density_image is None:
        raise ValueError(f"无法读取密度图: {density_map_path}")
    density_image = cv2.cvtColor(density_image, cv2.COLOR_BGR2RGB)
    
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
    
    # 为COCO标注创建颜色映射
    category_ids = set(ann['category_id'] for ann in coco_data['annotations'])
    color_map = {cat_id: np.random.rand(3,) for cat_id in category_ids}
    annotations_by_room = {}
    for ann in coco_data['annotations']:
        room_id = ann.get('room_id', 0)
        if room_id not in annotations_by_room:
            annotations_by_room[room_id] = []
        annotations_by_room[room_id].append(ann)
    
    # 创建7个子图
    fig = plt.figure(figsize=(24, 18))
    
    # 1. 点云密度图
    ax1 = fig.add_subplot(331)
    ax1.imshow(density_image)
    ax1.set_title('1. 点云密度图')
    ax1.axis('off')
    
    # 2. 原始房间边界
    ax2 = fig.add_subplot(332)
    box_x, box_y = zip(*box)
    ax2.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    for ann in annotations_filtered:
        rid = ann["room_id"]
        seg = ann["segmentation"][0]
        polygon = [(seg[i*2], seg[i*2+1]) for i in range(len(seg)//2)]
        x, y = zip(*polygon)
        x += (x[0],)
        y += (y[0],)
        ax2.plot(x, y, '-', color=room_color_map[rid], linewidth=1.5,
                label=f'房间 {rid}' if rid not in [l.get_label() for l in ax2.get_legend_handles_labels()[0]] else "")
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.set_title('2. 原始房间边界')
    
    # 3. 延伸至边界的墙面线段
    ax3 = fig.add_subplot(333)
    ax3.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    for seg in wall_segments:
        x, y = zip(*seg)
        ax3.plot(x, y, 'g-', linewidth=1.5, label='延伸后墙面线段' if ax3.get_legend_handles_labels()[1] == [] else "")
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.set_title('3. 延伸至边界的墙面线段')
    
    # 4. 约束Delaunay三角剖分结果
    ax4 = fig.add_subplot(334)
    ax4.triplot(triang, color='lightblue', linewidth=0.5)
    ax4.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)
    ax4.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.set_title('4. 约束Delaunay三角剖分结果')
    
    # 5. 按房间着色的三角剖分结果
    ax5 = fig.add_subplot(335)
    ax5.tripcolor(triang, adjusted_ids, cmap=cmap, alpha=0.7)
    ax5.triplot(triang, color='k', linewidth=0.5)
    ax5.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    for seg in wall_segments:
        x, y = zip(*seg)
        ax5.plot(x, y, 'g-', linewidth=1.5, label='墙面线段' if ax5.get_legend_handles_labels()[1] == [] else "")
    handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in cmap_colors[:-1]]
    labels = [f'房间 {rid}' for rid in sorted(room_polygons.keys())]
    handles.append(plt.Rectangle((0,0),1,1, facecolor=cmap_colors[-1]))
    labels.append('墙体/外部区域')
    ax5.legend(handles, labels, loc='best', fontsize=6)
    ax5.set_aspect('equal')
    ax5.set_title('5. 按房间着色的三角剖分结果')
    
    # 6. 点云密度图与可视化标注
    ax6 = fig.add_subplot(336)
    ax6.imshow(density_image)
    for room_id, anns in annotations_by_room.items():
        room_color_shift = random.uniform(0.3, 1.0)
        for ann in anns:
            cat_id = ann['category_id']
            base_color = color_map[cat_id] * room_color_shift
            
            is_line_annotation = (
                ann.get('area', 0) == 0.0 or 
                (len(ann.get('segmentation', [[]])[0]) < 6) or
                ann.get('bbox', [0,0,0,0])[3] == 0
            )
            
            if is_line_annotation:
                seg = ann['segmentation'][0]
                if len(seg) >= 4:
                    x1, y1, x2, y2 = seg[:4]
                    ax6.plot([x1, x2], [y1, y2], 
                            color=base_color, 
                            linewidth=2,
                            marker='o', 
                            markersize=5,
                            markerfacecolor='white',
                            markeredgecolor=base_color)
            else:
                for polygon_coords in ann['segmentation']:
                    if len(polygon_coords) < 6:
                        continue
                    poly = np.array(polygon_coords).reshape(-1, 2)
                    if len(poly) < 3 or calculate_area(poly) < 1.0:
                        continue
                    patch = Polygon(poly, closed=True, 
                                   facecolor=base_color, 
                                   alpha=0.4, 
                                   edgecolor='white', 
                                   linewidth=1)
                    ax6.add_patch(patch)
                bbox = ann['bbox']
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                    x, y, w, h = bbox
                    rect = plt.Rectangle((x, y), w, h, 
                                       linewidth=2,
                                       edgecolor=base_color, 
                                       facecolor='none',
                                       linestyle='--')
                    ax6.add_patch(rect)
    ax6.set_title('6. 点云密度图与可视化标注')
    ax6.axis('off')
    
    # 7. 点云密度图与可视化标注+三角剖分结果
    ax7 = fig.add_subplot(337)
    ax7.imshow(density_image)
    # 绘制三角剖分结果（半透明）
    ax7.tripcolor(triang, adjusted_ids, cmap=cmap, alpha=0.3)
    ax7.triplot(triang, color='k', linewidth=0.3, alpha=0.5)
    # 绘制标注
    for room_id, anns in annotations_by_room.items():
        room_color_shift = random.uniform(0.3, 1.0)
        for ann in anns:
            cat_id = ann['category_id']
            base_color = color_map[cat_id] * room_color_shift
            
            is_line_annotation = (
                ann.get('area', 0) == 0.0 or 
                (len(ann.get('segmentation', [[]])[0]) < 6) or
                ann.get('bbox', [0,0,0,0])[3] == 0
            )
            
            if is_line_annotation:
                seg = ann['segmentation'][0]
                if len(seg) >= 4:
                    x1, y1, x2, y2 = seg[:4]
                    ax7.plot([x1, x2], [y1, y2], 
                            color=base_color, 
                            linewidth=2,
                            marker='o', 
                            markersize=5,
                            markerfacecolor='white',
                            markeredgecolor=base_color)
            else:
                for polygon_coords in ann['segmentation']:
                    if len(polygon_coords) < 6:
                        continue
                    poly = np.array(polygon_coords).reshape(-1, 2)
                    if len(poly) < 3 or calculate_area(poly) < 1.0:
                        continue
                    patch = Polygon(poly, closed=True, 
                                   facecolor=base_color, 
                                   alpha=0.4, 
                                   edgecolor='white', 
                                   linewidth=1)
                    ax7.add_patch(patch)
                bbox = ann['bbox']
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                    x, y, w, h = bbox
                    rect = plt.Rectangle((x, y), w, h, 
                                       linewidth=2,
                                       edgecolor=base_color, 
                                       facecolor='none',
                                       linestyle='--')
                    ax7.add_patch(rect)
    ax7.set_title('7. 点云密度图与标注+三角剖分结果')
    ax7.axis('off')
    
    plt.tight_layout()
    plt.savefig('combined_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("所有可视化结果已保存为 combined_visualization.png")

# 使用示例
if __name__ == "__main__":
    scene_name = 'scene_000000'  # 替换为你的场景名称
    JSON_PATH = f"coco_with_scaled/sample0_256/anno/{scene_name}.json"
    DENSITY_MAP_PATH = f"coco_with_scaled/sample0_256/density_map/{scene_name}.png"
    
    visualize_combined_results(
        json_path=JSON_PATH,
        density_map_path=DENSITY_MAP_PATH,
        output_ply="combined_mesh.ply"
    )