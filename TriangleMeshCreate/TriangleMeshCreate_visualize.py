import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle  # pip install triangle
import json
import os
from matplotlib.colors import ListedColormap

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

# 加载数据
with open('coco_with_scaled/sample0_256/anno/scene_000001.json', 'r', encoding='utf-8') as f:
    coco_data = json.load(f)
annotations = coco_data.get('annotations', [])
# 过滤掉不需要的类别（保留有room_id的）
annotations = [ann for ann in annotations if 'room_id' in ann and ann.get('category_id') not in [0, 1]]


# 定义外包围盒
box = [
    (0, 0),
    (256, 0),
    (256, 256),
    (0, 256)
]
min_x, min_y = box[0]
max_x, max_y = box[2]

# 提取所有墙面线段并延伸至边界框
wall_segments = []
original_segments = []  # 保存原始线段用于可视化
all_points = []
room_polygons = {}  # 存储房间ID到多边形的映射 {room_id: [(x1,y1), (x2,y2), ...]}

for ann in annotations:
    room_id = ann["room_id"]
    seg = ann["segmentation"][0]  # coco格式: [x1,y1,x2,y2,...]
    polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
    
    # 保存房间多边形（用于后续三角形分配）
    if room_id not in room_polygons:
        room_polygons[room_id] = polygon
    else:
        room_polygons[room_id].extend(polygon)  # 处理可能的多段多边形
    
    # 提取多边形的边作为墙面线段
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        original_segments.append((p1, p2))  # 保存原始线段
        # 延伸线段至边界框
        extended = extend_segment_to_bbox((p1, p2), box)
        wall_segments.append(extended)
        all_points.extend(extended)

# 添加边界框的点
all_points.extend(box)
# 去重顶点
unique_points = []
seen = set()
for p in all_points:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in seen:
        seen.add(key)
        unique_points.append(p)

# 构建线段索引
segments = []
point_indices = { (round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(unique_points) }

# 去重线段（在构建segments前）
unique_segments = []
seen_seg = set()
for seg in wall_segments:
    # 标准化线段表示（按点索引排序，避免(a,b)和(b,a)被视为不同）
    p1, p2 = seg
    key = tuple(sorted([
        (round(p1[0], 6), round(p1[1], 6)),
        (round(p2[0], 6), round(p2[1], 6))
    ]))
    if key not in seen_seg:
        seen_seg.add(key)
        unique_segments.append(seg)
wall_segments = unique_segments  # 替换为去重后的线段

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

print("[DEBUG] 顶点数量:", len(unique_points))
print("[DEBUG] 线段数量:", len(segments))
print("[DEBUG] 房间数量:", len(room_polygons))
print("[DEBUG] 房间ID列表:", list(room_polygons.keys()))

# 检查索引是否越界
segments_np = np.array(segments, dtype=int)
max_idx = segments_np.max()
if max_idx >= len(unique_points):
    raise ValueError(f"❌ 线段索引 {max_idx} 超出了顶点范围（总顶点数={len(unique_points)}）")

# 构造输入字典
A = dict(
    vertices=np.array(unique_points, dtype=float),
    segments=segments_np
)

print("[DEBUG] 开始三角剖分...")
B = triangle.triangulate(A, 'p')
print("[DEBUG] 三角剖分完成!")

if 'triangles' not in B:
    print("❌ 三角剖分失败，未生成任何三角形")
else:
    # 可视化结果 - 创建多幅图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 原始线段图（按房间ID着色）
    ax1 = fig.add_subplot(221)
    # 绘制边界框
    box_x, box_y = zip(*box)
    ax1.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    # 为每个房间分配不同颜色绘制原始线段
    colors = plt.cm.tab10(np.linspace(0, 1, len(room_polygons)))
    room_color_map = {rid: colors[i] for i, rid in enumerate(room_polygons.keys())}
    for ann in annotations:
        rid = ann["room_id"]
        seg = ann["segmentation"][0]
        polygon = [(seg[i*2], seg[i*2+1]) for i in range(len(seg)//2)]
        x, y = zip(*polygon)
        x += (x[0],)  # 闭合多边形
        y += (y[0],)
        ax1.plot(x, y, '-', color=room_color_map[rid], linewidth=1.5, 
                label=f'房间 {rid}' if rid not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else "")
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title('1. 原始房间边界（按room_id着色）')
    
    # 2. 延伸后的线段图
    ax2 = fig.add_subplot(222)
    # 绘制边界框
    ax2.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    # 绘制延伸后的墙面线段
    for seg in wall_segments:
        x, y = zip(*seg)
        ax2.plot(x, y, 'g-', linewidth=1.5, label='延伸后墙面线段' if ax2.get_legend_handles_labels()[1] == [] else "")
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.set_title('2. 延伸至边界框的墙面线段')
    
    # 3. 三角剖分结果
    ax3 = fig.add_subplot(223)
    vertices = B['vertices']
    triangles = B['triangles']
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax3.triplot(triang, color='lightblue', linewidth=0.5)
    ax3.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)
    # 绘制边界框
    ax3.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.set_title('3. 约束Delaunay三角剖分结果')
    
    # 4. 按房间着色的三角剖分结果（基于room_id）
    ax4 = fig.add_subplot(224)
    # 将三角形分配到对应的房间
    triangle_room_ids = assign_triangles_to_rooms(triangles, vertices, room_polygons)
    
    # 创建颜色映射（使用与图1相同的颜色方案）
    cmap_colors = [room_color_map[rid] for rid in sorted(room_polygons.keys())]
    # 添加一个用于未分配区域（墙体/外部）的颜色
    cmap_colors.append([0.8, 0.8, 0.8])  # 灰色
    cmap = ListedColormap(cmap_colors)
    
    # 绘制着色的三角形
    # 调整ID以匹配颜色索引（未分配区域使用最后一个颜色）
    adjusted_ids = [rid if rid != -1 else len(room_polygons) for rid in triangle_room_ids]
    ax4.tripcolor(triang, adjusted_ids, cmap=cmap, alpha=0.7)
    # 绘制三角形边界
    ax4.triplot(triang, color='k', linewidth=0.5)
    # 绘制边界框
    ax4.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    # 绘制墙面线段
    for seg in wall_segments:
        x, y = zip(*seg)
        ax4.plot(x, y, 'g-', linewidth=1.5, label='墙面线段' if ax4.get_legend_handles_labels()[1] == [] else "")
    
    # 添加图例
    handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in cmap_colors[:-1]]
    labels = [f'房间 {rid}' for rid in sorted(room_polygons.keys())]
    handles.append(plt.Rectangle((0,0),1,1, facecolor=cmap_colors[-1]))
    labels.append('墙体/外部区域')
    ax4.legend(handles, labels, loc='best')
    
    ax4.set_aspect('equal')
    ax4.set_title(f'4. 按房间着色的三角剖分结果（共{len(room_polygons)}个房间）')
    
    plt.tight_layout()
    plt.show()

    # 保存为PLY文件（可包含房间ID信息）
    # save_triangulation_to_ply(vertices, triangles, filename='room_colored_mesh.ply', binary=True)
    print("[SUCCESS] 三角剖分结果已保存为 PLY 文件！")