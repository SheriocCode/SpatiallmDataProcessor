# 更加密集的划分三角网

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle
import json
import os

# ---------------------- 工具函数 ----------------------
def save_triangulation_to_ply(vertices, triangles, filename='output_mesh.ply', binary=True):
    """保存PLY文件（保持不变）"""
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

def round_point(p, decimals=6):
    """统一四舍五入点坐标"""
    return (round(p[0], decimals), round(p[1], decimals))

def line_intersection(seg1, seg2):
    """计算线段交点（带容差）"""
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / denom
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    u = u_num / denom

    if -1e-8 <= t <= 1+1e-8 and -1e-8 <= u <= 1+1e-8:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return round_point((x, y))
    return None

def extend_segment_to_bbox(segment, bbox):
    """延伸线段至边界框（保持不变）"""
    (x1, y1), (x2, y2) = segment
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[2]
    dx = x2 - x1
    dy = y2 - y1

    bbox_edges = [
        [(min_x, min_y), (max_x, min_y)],
        [(max_x, min_y), (max_x, max_y)],
        [(max_x, max_y), (min_x, max_y)],
        [(min_x, max_y), (min_x, min_y)]
    ]

    intersections = []
    for edge in bbox_edges:
        intersect = line_intersection(segment, edge)
        if intersect:
            intersections.append(intersect)

    if len(intersections) < 2:
        if abs(dx) < 1e-4:
            x = round((x1 + x2) / 2, 6)
            intersections.append((x, min_y))
            intersections.append((x, max_y))
        elif abs(dy) < 1e-4:
            y = round((y1 + y2) / 2, 6)
            intersections.append((min_x, y))
            intersections.append((max_x, y))
        else:
            m = dy / dx
            b = y1 - m * x1

            y_left = m * min_x + b
            y_right = m * max_x + b
            x_bottom = (min_y - b) / m
            x_top = (max_y - b) / m

            if min_y - 1e-8 <= y_left <= max_y + 1e-8:
                intersections.append(round_point((min_x, y_left)))
            if min_y - 1e-8 <= y_right <= max_y + 1e-8:
                intersections.append(round_point((max_x, y_right)))
            if min_x - 1e-8 <= x_bottom <= max_x + 1e-8:
                intersections.append(round_point((x_bottom, min_y)))
            if min_x - 1e-8 <= x_top <= max_x + 1e-8:
                intersections.append(round_point((x_top, max_y)))

    unique_intersections = []
    seen = set()
    for p in intersections:
        if p not in seen:
            seen.add(p)
            unique_intersections.append(p)
    intersections = unique_intersections

    if len(intersections) >= 2:
        dir_vec = (x2 - x1, y2 - y1)
        projections = []
        for p in intersections:
            proj = (p[0] - x1) * dir_vec[0] + (p[1] - y1) * dir_vec[1]
            projections.append((proj, p))
        projections.sort()
        return [projections[0][1], projections[-1][1]]
    elif len(intersections) == 1:
        p = intersections[0]
        if dx != 0:
            t = (min_x - p[0])/dx if dx < 0 else (max_x - p[0])/dx
        else:
            t = (min_y - p[1])/dy if dy < 0 else (max_y - p[1])/dy
        x = round(p[0] + t * dx, 6)
        y = round(p[1] + t * dy, 6)
        return [p, (x, y)]
    else:
        return [round_point((x1, y1)), round_point((x2, y2))]

def clean_segments(segments):
    """清理线段拓扑（保持不变）"""
    cleaned = []
    seen = set()

    for seg in segments:
        p1, p2 = seg
        if np.linalg.norm(np.array(p1) - np.array(p2)) < 1e-4:
            continue
        if p1 > p2:
            p1, p2 = p2, p1
        seg_key = (p1, p2)
        if seg_key not in seen:
            seen.add(seg_key)
            cleaned.append([p1, p2])
    
    return cleaned

def merge_colinear_segments(segments):
    """合并共线线段（保持不变）"""
    if not segments:
        return []
    
    merged = []
    segments.sort()
    
    current_p1, current_p2 = segments[0]
    for seg in segments[1:]:
        p1, p2 = seg
        vec1 = (current_p2[0] - current_p1[0], current_p2[1] - current_p1[1])
        vec2 = (p1[0] - current_p2[0], p1[1] - current_p2[1])
        cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        
        if abs(cross) < 1e-6 and np.linalg.norm(np.array(p1) - np.array(current_p2)) < 1e-4:
            current_p2 = p2
        else:
            merged.append([current_p1, current_p2])
            current_p1, current_p2 = p1, p2
    merged.append([current_p1, current_p2])
    
    return merged


# ---------------------- 主流程（核心优化部分） ----------------------
# 加载数据
with open('coco_with_scaled/sample0_256/anno/scene_000000_manual_repaired.json', 'r', encoding='utf-8') as f:
    coco_data = json.load(f)
annotations = coco_data.get('annotations', [])
annotations = [ann for ann in annotations if ann['category_id'] not in [0, 1]]

# 定义边界框
box = [(0, 0), (256, 0), (256, 256), (0, 256)]
min_x, min_y = box[0]
max_x, max_y = box[2]

# 提取并处理墙面线段
wall_segments = []
all_points = []

for ann in annotations:
    seg = ann["segmentation"][0]
    polygon = [(seg[i*2], seg[i*2+1]) for i in range(len(seg)//2)]
    n = len(polygon)
    for i in range(n):
        p1 = round_point(polygon[i])
        p2 = round_point(polygon[(i+1)%n])
        extended = extend_segment_to_bbox((p1, p2), box)
        wall_segments.append(extended)
        all_points.extend(extended)

# 清理线段拓扑
wall_segments = clean_segments(wall_segments)
wall_segments = merge_colinear_segments(wall_segments)

# 处理顶点
all_points.extend(box)
unique_points = []
seen = set()
for p in all_points:
    rp = round_point(p)
    if rp not in seen:
        seen.add(rp)
        unique_points.append(rp)

# 构建线段索引
point_indices = {rp: i for i, rp in enumerate(unique_points)}
segments = []
for seg in wall_segments:
    p1, p2 = seg
    idx1 = point_indices[p1]
    idx2 = point_indices[p2]
    segments.append((idx1, idx2))

# 添加边界框线段
box_rp = [round_point(p) for p in box]
box_indices = [point_indices[p] for p in box_rp]
n_box = len(box_indices)
for i in range(n_box):
    seg_key = (box_rp[i], box_rp[(i+1)%n_box])
    if seg_key not in [(round_point(seg[0]), round_point(seg[1])) for seg in wall_segments]:
        segments.append((box_indices[i], box_indices[(i+1)%n_box]))

# 输出清理后信息
print(f"[DEBUG] 清理后顶点数量: {len(unique_points)}")
print(f"[DEBUG] 清理后线段数量: {len(segments)}")

segments_np = np.array(segments, dtype=int)
max_idx = segments_np.max()
if max_idx >= len(unique_points):
    raise ValueError(f"❌ 线段索引 {max_idx} 超出顶点范围（总顶点数={len(unique_points)}）")

# ---------------------- 核心优化：控制三角形尺寸 ----------------------
# 计算合理的三角形最大面积（根据边界框大小动态调整）
bbox_area = (max_x - min_x) * (max_y - min_y)  # 边界框总面积
# 设定三角形数量预期（根据场景复杂度调整，例如预期100-200个三角形）
expected_triangles = 100  
max_area = bbox_area / expected_triangles  # 单个三角形最大面积

# 构造输入（添加面积约束参数）
A = dict(
    vertices=np.array(unique_points, dtype=float),
    segments=segments_np
)

print("[DEBUG] 开始三角剖分（带面积约束）...")
try:
    # 关键参数：
    # - 'p'：约束Delaunay剖分
    # - 'q15'：最小角不小于15度（避免过尖三角形）
    # - f{max_area}：三角形最大面积不超过max_area（核心控制密度）
    B = triangle.triangulate(A, f'pq15a{max_area}')
    print(f"[DEBUG] 三角剖分完成！生成三角形数量: {len(B.get('triangles', []))}")
except RuntimeError as e:
    print(f"[ERROR] 三角剖分失败: {e}")
    B = triangle.triangulate(A, f'pa{max_area}')  # 降级参数

if 'triangles' not in B or len(B['triangles']) == 0:
    print("❌ 三角剖分失败，未生成三角形")
else:
    # 可视化
    vertices = B['vertices']
    triangles = B['triangles']
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    plt.figure(figsize=(10, 10))
    plt.triplot(triang, color='lightblue', linewidth=0.5)
    plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)

    # 绘制边界框和墙面线段
    box_x, box_y = zip(*box)
    plt.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    for seg in wall_segments:
        x, y = zip(*seg)
        plt.plot(x, y, 'g-', linewidth=1.5)

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title(f'带面积约束的三角剖分（约{len(triangles)}个三角形）')
    plt.show()

    # 保存结果
    save_triangulation_to_ply(vertices, triangles, filename='optimized_wall_mesh.ply', binary=True)
    print("[SUCCESS] 优化后的三角剖分结果已保存！")