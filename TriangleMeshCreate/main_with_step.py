import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle  # pip install triangle
import json
import os

def save_triangulation_to_ply(vertices, triangles, filename='output_mesh.ply', binary=True):
    """将三角剖分结果保存为 PLY 文件格式"""
    if vertices.ndim == 2 and vertices.shape[1] == 2:
        z = np.zeros((vertices.shape[0], 1), dtype=vertices.dtype)
        vertices = np.hstack([vertices, z])  # 变成 (N, 3)
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
        m = dy / dx  # 斜率
        b = y1 - m * x1  # 截距

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
    
    return segment  # 无法延伸的情况


def check_manhattan_orientation(segment, tolerance=1e-3):
    """
    改进的曼哈顿假设检测：更准确地识别水平和垂直线段
    返回：(is_horizontal, is_vertical, angle, orientation)
    orientation: 'horizontal', 'vertical', 'diagonal'
    """
    (x1, y1), (x2, y2) = segment
    dx = x2 - x1
    dy = y2 - y1
    
    # 计算线段长度（避免零长度线段）
    length = np.hypot(dx, dy)
    if length < tolerance:
        return (False, False, 0.0, 'zero_length')  # 零长度线段
    
    # 检查是否水平（y坐标几乎不变）
    is_horizontal = abs(dy) < tolerance
    
    # 检查是否垂直（x坐标几乎不变）
    is_vertical = abs(dx) < tolerance
    
    # 计算角度（对垂直线做特殊处理）
    if is_vertical:
        angle = 90.0  # 垂直线角度为90度
        orientation = 'vertical'
    elif is_horizontal:
        angle = 0.0   # 水平线角度为0度
        orientation = 'horizontal'
    else:
        # 计算与水平线的夹角（度）
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        orientation = 'diagonal'
    
    return (is_horizontal, is_vertical, angle, orientation)


def visualize_step(points, segments=None, title="调试可视化", bbox=None, 
                  raw_segments=None, problematic_segments=None, annotations_data=None):
    """可视化某一步的结果，特别标记问题线段"""
    plt.figure(figsize=(12, 12))
    
    # 绘制边界框
    if bbox:
        box_x, box_y = zip(*bbox)
        plt.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    
    # 绘制原始线段
    if raw_segments:
        for seg in raw_segments:
            x, y = zip(*seg)
            plt.plot(x, y, 'y-', linewidth=1.5, label='原始线段' if raw_segments.index(seg) == 0 else "")
    
    # 绘制正常线段
    if segments:
        for i, seg in enumerate(segments):
            # 检查是否是问题线段
            is_problem = False
            if problematic_segments and i < len(problematic_segments):
                is_problem = problematic_segments[i]
                
            if isinstance(seg[0], tuple):  # 直接是点对
                x, y = zip(*seg)
            else:  # 是索引
                p1 = points[seg[0]]
                p2 = points[seg[1]]
                x, y = [p1[0], p2[0]], [p1[1], p2[1]]
            
            # 线段颜色编码：水平(绿)、垂直(蓝)、问题(红)
            if is_problem:
                color = 'r-'
                linewidth = 2.5
                label = '非曼哈顿线段' if '非曼哈顿线段' not in plt.gca().get_legend_handles_labels()[1] else ""
            else:
                if annotations_data and i < len(annotations_data):
                    orientation = annotations_data[i]['orientation']
                    if orientation == 'horizontal':
                        color = 'g-'
                        label = '水平线段' if '水平线段' not in plt.gca().get_legend_handles_labels()[1] else ""
                    else:  # vertical
                        color = 'b-'
                        label = '垂直线段' if '垂直线段' not in plt.gca().get_legend_handles_labels()[1] else ""
                else:
                    color = 'g-'
                    label = ""
                linewidth = 1.5
            
            plt.plot(x, y, color, linewidth=linewidth, label=label)
            
            # 为问题线段添加标注信息
            if is_problem and annotations_data and i < len(annotations_data):
                ann_data = annotations_data[i]
                # 线段中点
                mid_x = (x[0] + x[1]) / 2
                mid_y = (y[0] + y[1]) / 2
                plt.text(mid_x, mid_y, 
                         f"room:{ann_data['room_id']}\nid:{ann_data['id']}\n角度:{ann_data['angle']:.4f}°",
                         fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
    
    # 绘制点
    if points:
        x, y = zip(*points)
        plt.scatter(x, y, color='blue', s=10, label='顶点')
    
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# 从文件加载 S3D.json
try:
    with open('coco/scene_000000.json', 'r', encoding='utf-8') as f:
        s3d_data = json.load(f)
    print("[INFO] 成功加载JSON文件")
except Exception as e:
    print(f"[ERROR] 加载JSON文件失败: {e}")
    exit(1)

# 提取标注数据
annotations = s3d_data.get('annotations', [])
print(f"[DEBUG] 原始标注数量: {len(annotations)}")

# 排除指定category_id的annotation
filtered_annotations = [ann for ann in annotations if ann['category_id'] not in [0, 1]]
print(f"[DEBUG] 过滤后标注数量: {len(filtered_annotations)}")

# 定义外包围盒
box = [
    (0, 0),
    (256, 0),
    (256, 256),
    (0, 256)
]
min_x, min_y = box[0]
max_x, max_y = box[2]

# 提取原始多边形线段用于可视化
raw_polygons = []
for ann in filtered_annotations:
    seg = ann["segmentation"][0]
    polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    raw_polygons.append(polygon)

# 可视化1: 原始标注数据
visualize_step(
    points=[p for poly in raw_polygons for p in poly],
    segments=raw_polygons,
    title="1. 原始标注数据",
    bbox=box
)

# 提取所有墙面线段并检查曼哈顿方向
wall_segments = []
raw_wall_segments = []  # 保存原始线段
all_points = []
problematic_segments = []  # 标记哪些线段不符合曼哈顿假设
annotations_data = []  # 存储线段对应的标注信息

print("\n[DEBUG] 开始检查线段方向是否符合曼哈顿假设...")
for ann in filtered_annotations:
    ann_id = ann['id']
    room_id = ann.get('room_id', -1)  # 获取房间ID
    seg = ann["segmentation"][0]
    raw_polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
    
    # 处理闭合点
    if len(raw_polygon) >= 3 and raw_polygon[0] == raw_polygon[-1]:
        polygon = raw_polygon[:-1]
    else:
        polygon = raw_polygon
    
    # 检查每条边
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        raw_segment = [p1, p2]
        raw_wall_segments.append(raw_segment)
        
        # 检查是否符合曼哈顿假设（改进版）
        is_horizontal, is_vertical, angle, orientation = check_manhattan_orientation(raw_segment)
        is_problem = not (is_horizontal or is_vertical)
        
        # 记录线段信息（包括方向类型）
        seg_info = {
            'room_id': room_id,
            'id': ann_id,
            'angle': angle,
            'orientation': orientation,
            'segment': raw_segment
        }
        
        # 记录问题线段信息
        if is_problem:
            print(f"[WARNING] 非曼哈顿线段 - room_id: {room_id}, annotation_id: {ann_id}, "
                  f"线段: {p1} -> {p2}, 角度: {angle:.4f}°")
            problematic_segments.append(True)
        else:
            print(f"[INFO] 曼哈顿线段 - room_id: {room_id}, annotation_id: {ann_id}, "
                  f"类型: {orientation}, 线段: {p1} -> {p2}, 角度: {angle:.4f}°")
            problematic_segments.append(False)
        
        annotations_data.append(seg_info)
        
        # 延伸线段
        extended = extend_segment_to_bbox((p1, p2), box)
        wall_segments.append(extended)
        all_points.extend(extended)

# 可视化2: 线段延伸结果与曼哈顿检查
visualize_step(
    points=all_points,
    segments=wall_segments,
    title="2. 线段延伸结果与曼哈顿检查（绿:水平, 蓝:垂直, 红:非曼哈顿）",
    bbox=box,
    raw_segments=raw_wall_segments,
    problematic_segments=problematic_segments,
    annotations_data=annotations_data
)

# 添加边界框的点并去重
all_points.extend(box)
print(f"[DEBUG] 延伸后总点数（含重复）: {len(all_points)}")

unique_points = []
seen = set()
for p in all_points:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in seen:
        seen.add(key)
        unique_points.append(p)

print(f"[DEBUG] 去重后顶点数量: {len(unique_points)}")

# 可视化3: 去重后的顶点
visualize_step(
    points=unique_points,
    title="3. 去重后的顶点",
    bbox=box
)

# 构建线段索引
segments = []
point_indices = { (round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(unique_points) }

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

print("[DEBUG] 线段数量:", len(segments))

# 可视化4: 基于索引的线段
visualize_step(
    points=unique_points,
    segments=segments,
    title="4. 基于索引的线段",
    bbox=box
)

# 检查索引是否越界
segments_np = np.array(segments, dtype=int)
max_idx = segments_np.max()
if max_idx >= len(unique_points):
    raise ValueError(f"❌ 线段索引 {max_idx} 超出了顶点范围（总顶点数={len(unique_points)}）")

# 构造输入字典并进行三角剖分
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
    # 可视化5: 最终三角剖分结果
    vertices = B['vertices']
    triangles = B['triangles']

    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    plt.figure(figsize=(10, 10))
    plt.triplot(triang, color='lightblue', linewidth=0.5, label='三角形网格')
    plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2, label='顶点')

    # 绘制边界框
    box_x, box_y = zip(*box)
    plt.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')

    # 绘制延伸后的墙面线段，按类型着色
    for i, seg in enumerate(wall_segments):
        x, y = zip(*seg)
        if problematic_segments[i]:
            color = 'r-'
            label = '非曼哈顿线段' if '非曼哈顿线段' not in plt.gca().get_legend_handles_labels()[1] else ""
        else:
            orientation = annotations_data[i]['orientation']
            if orientation == 'horizontal':
                color = 'g-'
                label = '水平线段' if '水平线段' not in plt.gca().get_legend_handles_labels()[1] else ""
            else:
                color = 'b-'
                label = '垂直线段' if '垂直线段' not in plt.gca().get_legend_handles_labels()[1] else ""
        plt.plot(x, y, color, linewidth=1.5, label=label)

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('5. 最终三角剖分结果')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 保存为PLY文件
    save_triangulation_to_ply(vertices, triangles, filename='extended_wall_mesh.ply', binary=True)
    print("[SUCCESS] 三角剖分结果已保存为 PLY 文件！")