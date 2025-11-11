import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle
import json
import os
from shapely.geometry import Point, Polygon
from shapely import wkt

def is_point_in_polygon(point, polygon_coords):
    """精确判断点是否在多边形内（支持凹多边形）"""
    point_geom = Point(point)
    poly_geom = Polygon(polygon_coords)
    return poly_geom.contains(point_geom) or poly_geom.touches(point_geom)

def assign_room_ids_to_triangles(triangles, vertices, annotations):
    """
    为每个三角形分配房间ID
    规则：重心落在哪个原始房间多边形内，就归属该房间
    """
    triangle_rooms = np.full(len(triangles), -1, dtype=int)  # -1表示未分配
    
    # 预处理房间多边形
    room_polygons = []
    room_ids = []
    for ann in annotations:
        seg = ann["segmentation"][0]
        polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
        room_polygons.append(polygon)
        room_ids.append(ann.get('id', ann.get('category_id', -1)))
    
    # 为每个三角形分配房间
    for i, tri in enumerate(triangles):
        # 计算重心
        center = vertices[tri].mean(axis=0)
        # 检查属于哪个房间
        for j, polygon in enumerate(room_polygons):
            if is_point_in_polygon(center, polygon):
                triangle_rooms[i] = room_ids[j]
                break
    
    return triangle_rooms

def generate_room_colors(room_ids):
    """为每个房间生成唯一颜色"""
    unique_rooms = np.unique(room_ids)
    # 使用tab20颜色映射，最多支持20个房间
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_rooms), 20)))
    room_color_map = {}
    for i, room_id in enumerate(unique_rooms):
        if room_id == -1:
            room_color_map[room_id] = np.array([0.7, 0.7, 0.7, 1.0])  # 灰色：未分配
        else:
            room_color_map[room_id] = colors[i % 20]
    return room_color_map

def save_colored_ply(vertices, triangles, triangle_rooms, room_color_map, 
                     filename='colored_mesh.ply', binary=True):
    """保存带房间颜色的PLY文件"""
    if vertices.ndim == 2 and vertices.shape[1] == 2:
        vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1))])
    
    num_vertices = len(vertices)
    num_faces = len(triangles)
    
    # 准备面颜色
    face_colors = np.zeros((num_faces, 3), dtype=np.uint8)
    for i, room_id in enumerate(triangle_rooms):
        color = room_color_map.get(room_id, [0.5, 0.5, 0.5, 1])
        face_colors[i] = (np.array(color[:3]) * 255).astype(np.uint8)
    
    # 构建PLY头
    header_lines = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    header = '\n'.join(header_lines) + '\n'
    
    with open(filename, 'wb' if binary else 'w') as f:
        if binary:
            f.write(header.encode('ascii'))
            f.write(vertices.astype(np.float32).tobytes())
            for i, face in enumerate(triangles):
                f.write(np.uint8(3).tobytes())
                f.write(face.astype(np.int32).tobytes())
                f.write(face_colors[i].tobytes())
        else:
            f.write(header)
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for i, face in enumerate(triangles):
                f.write(f"3 {face[0]} {face[1]} {face[2]} "
                       f"{face_colors[i][0]} {face_colors[i][1]} {face_colors[i][2]}\n")
    print(f"[INFO] 彩色网格已保存: {os.path.abspath(filename)}")

# ==================== 原始函数保持不变 ====================
def line_intersection(seg1, seg2):
    """计算两条线段的交点"""
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None

def extend_segment_to_bbox(segment, bbox):
    """将线段延伸至边界框"""
    (x1, y1), (x2, y2) = segment
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[2]
    
    # 特殊处理垂直/水平线
    if abs(x2 - x1) < 1e-6:  # 垂直线
        return [(x1, min_y), (x1, max_y)]
    if abs(y2 - y1) < 1e-6:  # 水平线
        return [(min_x, y1), (max_x, y1)]
    
    # 计算与边界框的交点
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    intersections = []
    # 与左右边界交点
    y_left = m * min_x + b
    if min_y <= y_left <= max_y:
        intersections.append((min_x, y_left))
    y_right = m * max_x + b
    if min_y <= y_right <= max_y:
        intersections.append((max_x, y_right))
    # 与上下边界交点
    x_bottom = (min_y - b) / m
    if min_x <= x_bottom <= max_x:
        intersections.append((x_bottom, min_y))
    x_top = (max_y - b) / m
    if min_x <= x_top <= max_x:
        intersections.append((x_top, max_y))
    
    # 取两个最远的点
    if len(intersections) >= 2:
        # 计算距离并排序
        intersections.sort(key=lambda p: (p[0]-x1)**2 + (p[1]-y1)**2)
        return [intersections[0], intersections[-1]]
    return segment
# ==================== 主流程 ====================

# 1. 加载数据
with open('s3d/00002.json', 'r', encoding='utf-8') as f:
    s3d_data = json.load(f)

annotations = [ann for ann in s3d_data['annotations'] if ann['category_id'] not in [16, 17]]

# 2. 定义边界框
bbox = [(0, 0), (256, 0), (256, 256), (0, 256)]
min_x, min_y = 0, 0
max_x, max_y = 256, 256

# 3. 提取墙面并延伸
wall_segments = []
all_points = []
segment_room_ids = []  # 记录每条线段的房间ID

for ann in annotations:
    seg = ann["segmentation"][0]
    polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
    room_id = ann.get('id', ann.get('category_id', -1))
    
    for i in range(len(polygon)):
        p1, p2 = polygon[i], polygon[(i + 1) % len(polygon)]
        extended = extend_segment_to_bbox((p1, p2), bbox)
        wall_segments.append(extended)
        all_points.extend(extended)
        segment_room_ids.append(room_id)

all_points.extend(bbox)
# 去重顶点
unique_points = []
seen = set()
for p in all_points:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in seen:
        seen.add(key)
        unique_points.append(p)

# 4. 构建线段索引
segments = []
point_indices = { (round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(unique_points) }

for idx, seg in enumerate(wall_segments):
    p1, p2 = seg
    idx1 = point_indices[(round(p1[0], 6), round(p1[1], 6))]
    idx2 = point_indices[(round(p2[0], 6), round(p2[1], 6))]
    segments.append((idx1, idx2))

# 添加边界框边（房间ID设为-1）
box_indices = [point_indices[(round(p[0], 6), round(p[1], 6))] for p in bbox]
for i in range(len(box_indices)):
    segments.append((box_indices[i], box_indices[(i + 1) % len(box_indices)]))
    segment_room_ids.append(-1)

# 5. 执行约束Delaunay三角剖分
A = dict(vertices=np.array(unique_points, dtype=float), segments=np.array(segments, dtype=int))
B = triangle.triangulate(A, 'p')

if 'triangles' not in B:
    print("❌ 三角剖分失败")
    exit()

vertices = B['vertices']
triangles = B['triangles']

# 6. 为三角形分配房间ID
print("[INFO] 正在分配房间ID到三角形...")
triangle_rooms = assign_room_ids_to_triangles(triangles, vertices, annotations)
room_color_map = generate_room_colors(triangle_rooms)

# 7. 三阶段可视化
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# 子图1：原始房间布局
ax1 = axes[0]
ax1.set_title('原始房间布局', fontsize=12, fontweight='bold')
for ann in annotations:
    seg = ann["segmentation"][0]
    polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
    room_id = ann.get('id', ann.get('category_id', -1))
    color = room_color_map.get(room_id, [0.5, 0.5, 0.5, 1])
    poly = plt.Polygon(polygon, facecolor=color[:3], alpha=0.6, edgecolor='black', linewidth=1.5)
    ax1.add_patch(poly)
ax1.set_aspect('equal')
ax1.set_xlim(-10, 266)
ax1.set_ylim(-10, 266)
ax1.grid(True, alpha=0.3)

# 子图2：延伸后的墙面线段
ax2 = axes[1]
ax2.set_title('延伸后的墙面线段', fontsize=12, fontweight='bold')
for idx, seg in enumerate(wall_segments):
    room_id = segment_room_ids[idx]
    color = room_color_map.get(room_id, [0.5, 0.5, 0.5, 1])
    x, y = zip(*seg)
    ax2.plot(x, y, '-', color=color[:3], linewidth=2, alpha=0.8)
# 绘制边界框
box_x, box_y = zip(*bbox)
ax2.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=3, label='边界框')
ax2.legend()
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# 子图3：按房间着色的三角剖分
ax3 = axes[2]
ax3.set_title('按房间着色的三角剖分', fontsize=12, fontweight='bold')
# 绘制所有顶点
ax3.plot(vertices[:, 0], vertices[:, 1], 'o', color='black', markersize=1, alpha=0.3)

# 按房间绘制三角形
for room_id in np.unique(triangle_rooms):
    mask = triangle_rooms == room_id
    color = room_color_map.get(room_id, [0.7, 0.7, 0.7, 1])
    room_tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles[mask])
    ax3.triplot(room_tri, color=color[:3], linewidth=0.7, alpha=0.9)

ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. 保存彩色PLY文件
save_colored_ply(vertices, triangles, triangle_rooms, room_color_map, 
                 filename='room_colored_mesh.ply', binary=True)

print(f"[SUCCESS] 处理完成！")
print(f"   - 顶点数: {len(vertices)}")
print(f"   - 三角形数: {len(triangles)}")
print(f"   - 房间数: {len([r for r in np.unique(triangle_rooms) if r != -1])}")