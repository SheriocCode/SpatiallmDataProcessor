# 现在除了房间三角形和外部三角形外，现在增加一种特殊三角形，用橙色标记。
# 如果一个三角形和共享该三角形最大内角的对边的相邻三角形都满足顶点都在房间边上（每个顶点可以分别在不同房间的边上），
# 并且该三角形和对边三角形都不属于房间内部三角形，则认定该三角形和对边相邻三角形均为特殊三角形。
# 其中点“在房间上”定义为：该点在房间多边形边上


import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle  # pip install triangle
import json
import os
from matplotlib.colors import ListedColormap
import logging
from utils.log_util import init_logger
from plyfile import PlyData, PlyElement  # 新增：用于读取PLY文件
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely.errors import TopologicalError

init_logger('rule_output/rule_test.log')

# ply 携带 room_id 和 inner_wall
def save_ply_with_face_attrs(vertices, triangles, triangle_room_ids, is_special, 
                             filename='output_mesh.ply', binary=True):
    """兼容无效room_id的PLY保存函数，处理-1等超出uchar范围的值"""
    # 顶点维度处理（2D转3D）
    if vertices.ndim == 2 and vertices.shape[1] == 2:
        z = np.zeros((vertices.shape[0], 1), dtype=vertices.dtype)
        vertices = np.hstack([vertices, z])
    elif vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices 必须是 N×2 或 N×3 的数组，但得到了 {vertices.shape}")

    num_vertices = vertices.shape[0]
    num_faces = triangles.shape[0]

    # 验证面属性数据长度是否匹配
    if len(triangle_room_ids) != num_faces:
        raise ValueError(f"triangle_room_ids长度({len(triangle_room_ids)})与面数量({num_faces})不匹配")
    if len(is_special) != num_faces:
        raise ValueError(f"is_special长度({len(is_special)})与面数量({num_faces})不匹配")

    # ----------------------
    # 关键修复：处理无效room_id（-1映射为255，因为uchar范围0-255，255作为"无房间"标记）
    # ----------------------
    room_ids = np.array(triangle_room_ids, dtype=np.int32)  # 先转为int32避免溢出
    # 映射规则：-1 → 255（无房间），其他值保持不变（需确保原始有效room_id≤254，留255给无效值）
    room_ids[room_ids == -1] = 255
    # 验证映射后的值是否在uchar范围内（0-255）
    if np.any(room_ids < 0) or np.any(room_ids > 255):
        logging.warning(f"[WARNING] 部分room_id超出uchar范围，已自动裁剪到0-255")
        room_ids = np.clip(room_ids, 0, 255)  # 强制裁剪，避免崩溃
    room_ids = room_ids.astype(np.uint8)  # 最终转为uchar类型

    # is_inner_wall确保为0/1的uchar
    is_inner_wall = np.array(is_special, dtype=np.uint8)
    is_inner_wall = np.clip(is_inner_wall, 0, 1)  # 防止非0/1值

    # PLY头部（包含新增属性）
    header_lines = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        "property uchar room_id",             # 0-254=有效房间ID，255=无房间（原-1）
        "property uchar is_inner_wall",       # 0=非内墙，1=内墙
        "end_header"
    ]

    header = '\n'.join(header_lines) + '\n'

    with open(filename, 'wb' if binary else 'w') as f:
        if binary:
            f.write(header.encode('ascii'))
        else:
            f.write(header)

        # 写入顶点数据
        if binary:
            vertices_flat = vertices.astype(np.float32).flatten()
            f.write(vertices_flat.tobytes())
        else:
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # 写入面数据（包含新增属性）
        if binary:
            for i in range(num_faces):
                face = triangles[i]
                # 面数据格式：[顶点数量(uchar) + 顶点索引(int32×3) + room_id(uchar) + is_inner_wall(uchar)]
                face_header = np.array([3], dtype=np.uint8)
                face_indices = face.astype(np.int32)
                face_data = np.concatenate([
                    face_header,
                    face_indices.view(np.uint8),
                    np.array([room_ids[i]], dtype=np.uint8),
                    np.array([is_inner_wall[i]], dtype=np.uint8)
                ])
                f.write(face_data.tobytes())
        else:
            for i in range(num_faces):
                face = triangles[i]
                f.write(f"3 {face[0]} {face[1]} {face[2]} {room_ids[i]} {is_inner_wall[i]}\n")

    logging.info(f"[INFO] 带面属性的PLY文件已保存: {os.path.abspath(filename)}")

# 提取scene多边形
def extract_enclosed_regions(vertices, triangles, triangle_room_ids, is_special):
    """
    提取由房间内部三角形和特殊内墙三角形构成的封闭区域外边界
    
    参数:
        vertices: 顶点坐标数组
        triangles: 三角形索引数组
        triangle_room_ids: 三角形所属房间ID
        is_special: 三角形是否为特殊三角形的标记数组
        
    返回:
        外边界多边形列表，每个多边形为点坐标列表
    """
    # 收集所有属于房间内部或特殊三角形的几何对象
    polygons = []
    for i in range(len(triangles)):
        # 房间内部三角形或特殊三角形
        if triangle_room_ids[i] != -1 or is_special[i]:
            # 获取三角形三个顶点坐标
            p1 = vertices[triangles[i][0]]
            p2 = vertices[triangles[i][1]]
            p3 = vertices[triangles[i][2]]
            
            # 创建三角形多边形
            tri_poly = sg.Polygon([p1, p2, p3, p1])  # 闭合多边形
            polygons.append(tri_poly)
    
    if not polygons:
        return []
    
    try:
        # 合并所有三角形为一个或多个连续区域
        union = unary_union(polygons)
        
        # 提取外边界
        outer_boundaries = []
        
        if isinstance(union, sg.MultiPolygon):
            # 处理多个分离的区域
            for poly in union.geoms:
                # 只保留面积最大的外环
                if poly.exterior:
                    outer_boundaries.append(poly.exterior)
        elif isinstance(union, sg.Polygon):
            # 处理单个区域
            if union.exterior:
                outer_boundaries.append(union.exterior)
        
        # 转换为点列表格式
        result = []
        for boundary in outer_boundaries:
            # 提取坐标并去除最后一个点（与第一个点重复）
            coords = list(boundary.coords)[:-1]
            # 转换为普通列表格式
            polygon = [list(coord)[:2] for coord in coords]  # 只保留x,y坐标
            result.append(polygon)
            
        return result
        
    except TopologicalError as e:
        logging.error(f"提取封闭区域时发生拓扑错误: {e}")
        return []

# 新增：从PLY文件加载三角剖分结果
def load_triangulation_from_ply(filename):
    """从PLY文件加载三角剖分数据，与save_triangulation_to_ply保持互逆"""
    with open(filename, 'rb') as f:
        # 读取头部信息
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        
        # 解析头部信息
        num_vertices = 0
        num_faces = 0
        is_binary = False
        
        for line in header:
            if line.startswith('format'):
                is_binary = 'binary' in line
            elif line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
        
        # 读取顶点数据
        vertices = []
        if is_binary:
            # 二进制格式读取
            vertex_data = np.fromfile(f, dtype=np.float32, count=num_vertices * 3)
            vertices = vertex_data.reshape((num_vertices, 3))
        else:
            # ASCII格式读取
            for _ in range(num_vertices):
                x, y, z = map(float, f.readline().decode('ascii').split())
                vertices.append([x, y, z])
            vertices = np.array(vertices, dtype=np.float32)
        
        # 读取三角形数据
        triangles = []
        if is_binary:
            # 二进制格式读取
            face_data = np.fromfile(f, dtype=np.uint8, count=num_faces * (1 + 3 * 4))
            ptr = 0
            for _ in range(num_faces):
                # 每个面开头是顶点数量(3)，然后是3个顶点索引
                if face_data[ptr] != 3:
                    raise ValueError("只支持三角形面的PLY文件")
                ptr += 1
                # 顶点索引是int32类型
                indices = np.frombuffer(face_data[ptr:ptr+12], dtype=np.int32)
                triangles.append(indices)
                ptr += 12
        else:
            # ASCII格式读取
            for _ in range(num_faces):
                parts = list(map(int, f.readline().decode('ascii').split()))
                if parts[0] != 3:
                    raise ValueError("只支持三角形面的PLY文件")
                triangles.append(parts[1:4])
        
        triangles = np.array(triangles, dtype=np.int32)
        
        # 如果原始是2D数据，返回时去掉z轴
        if np.allclose(vertices[:, 2], 0):
            vertices = vertices[:, :2]
            
        return vertices, triangles

# 新增：判断点是否在多边形边上（点到线段的距离小于阈值）
def point_on_polygon_edge(point, polygon, epsilon=1e-6):
    x, y = point
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        x1, y1 = p1
        x2, y2 = p2
        
        # 计算点到线段的距离
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) > epsilon:
            continue  # 不在这条线段上
        
        # 检查点是否在线段的 bounding box 内
        min_x = min(x1, x2) - epsilon
        max_x = max(x1, x2) + epsilon
        min_y = min(y1, y2) - epsilon
        max_y = max(y1, y2) + epsilon
        
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True
    return False

# 新增：获取点所在的所有房间（点在房间边上）
def get_point_rooms(point, room_polygons):
    rooms = []
    for room_id, polygon in room_polygons.items():
        if point_on_polygon_edge(point, polygon):
            rooms.append(room_id)
    return rooms

# 新增：找到共享边的相邻三角形
def find_adjacent_triangles(triangles):
    edge_map = {}
    adjacency = [[] for _ in range(len(triangles))]
    
    for tri_idx, tri in enumerate(triangles):
        # 生成三角形的三条边（按排序的顶点索引表示）
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        
        for edge in edges:
            if edge in edge_map:
                # 找到共享这条边的另一个三角形
                other_idx = edge_map[edge]
                adjacency[tri_idx].append((other_idx, edge))
                adjacency[other_idx].append((tri_idx, edge))
                del edge_map[edge]  # 避免重复处理
            else:
                edge_map[edge] = tri_idx
    
    return adjacency

# 新增：计算三角形最大内角的对边
def find_max_angle_opposite_edge(vertices, triangle):
    p0 = vertices[triangle[0]]
    p1 = vertices[triangle[1]]
    p2 = vertices[triangle[2]]
    
    # 计算三边长
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p0 - p2)
    c = np.linalg.norm(p0 - p1)
    
    # 计算三个角度（使用余弦定理）
    angles = []
    if b > 0 and c > 0:
        angle0 = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
        angles.append(angle0)
    else:
        angles.append(0)
        
    if a > 0 and c > 0:
        angle1 = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
        angles.append(angle1)
    else:
        angles.append(0)
        
    if a > 0 and b > 0:
        angle2 = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
        angles.append(angle2)
    else:
        angles.append(0)
    
    # 找到最大角的索引
    max_angle_idx = np.argmax(angles)
    
    # 返回最大角的对边
    opposite_edges = [
        tuple(sorted([triangle[1], triangle[2]])),  # 角0的对边
        tuple(sorted([triangle[0], triangle[2]])),  # 角1的对边
        tuple(sorted([triangle[0], triangle[1]]))   # 角2的对边
    ]
    
    return opposite_edges[max_angle_idx]

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

    logging.info(f"[INFO] 三角网格已保存为 PLY 文件: {os.path.abspath(filename)}")


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

def process_and_visualize_scene(json_path, output_image_name, ply_path=None):
    # 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
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

    # 提取房间多边形（用于后续三角形分配）
    room_polygons = {}  # 存储房间ID到多边形的映射 {room_id: [(x1,y1), (x2,y2), ...]}
    for ann in annotations:
        room_id = ann["room_id"]
        seg = ann["segmentation"][0]  # coco格式: [x1,y1,x2,y2,...]
        polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
        
        # 保存房间多边形
        if room_id not in room_polygons:
            room_polygons[room_id] = polygon
        else:
            room_polygons[room_id].extend(polygon)  # 处理可能的多段多边形

    # 尝试从PLY文件加载三角剖分结果
    vertices = None
    triangles = None
    if ply_path and os.path.exists(ply_path):
        vertices, triangles = load_triangulation_from_ply(ply_path)
    
    # 提取所有墙面线段并延伸至边界框
    wall_segments = []
    original_segments = []  # 保存原始线段用于可视化
    all_points = []

    for ann in annotations:
        room_id = ann["room_id"]
        seg = ann["segmentation"][0]  # coco格式: [x1,y1,x2,y2,...]
        polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
        
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

    logging.info(f"[DEBUG] 顶点数量: {len(unique_points)}")
    logging.info(f"[DEBUG] 线段数量: {len(segments)}")
    logging.info(f"[DEBUG] 房间数量: {len(room_polygons)}")
    logging.info(f"[DEBUG] 房间ID列表: { list(room_polygons.keys())}")

    # 检查索引是否越界
    segments_np = np.array(segments, dtype=int)
    max_idx = segments_np.max()
    if max_idx >= len(unique_points):
        raise ValueError(f"❌ 线段索引 {max_idx} 超出了顶点范围（总顶点数={len(unique_points)}）")

    # 如果加载失败或没有指定PLY文件，则进行三角剖分
    # if vertices is None or triangles is None:
    #     logging.info("[DEBUG] 未找到有效PLY文件，进行三角剖分...")
    #     # 构造输入字典
    #     A = dict(
    #         vertices=np.array(unique_points, dtype=float),
    #         segments=segments_np
    #     )

    #     logging.info("[DEBUG] 开始三角剖分...")
    #     B = triangle.triangulate(A, 'p')
    #     logging.info("[DEBUG] 三角剖分完成!")

    #     if 'triangles' not in B:
    #         logging.error("❌ 三角剖分失败，未生成任何三角形")
    #         return
    #     else:
    #         vertices = B['vertices']
    #         triangles = B['triangles']
            
    #         # 如果指定了PLY路径，保存结果
    #         if ply_path:
    #             save_triangulation_to_ply(vertices, triangles, filename=ply_path, binary=True)

    # 计算三角形房间归属
    triangle_room_ids = assign_triangles_to_rooms(triangles, vertices, room_polygons)
    
    # 新增：识别特殊三角形
    logging.info("[DEBUG] 开始识别特殊三角形...")
    # 预处理：获取每个顶点所在的房间（点在房间边上）
    vertex_rooms = []
    for v in vertices:
        rooms = get_point_rooms(v, room_polygons)
        vertex_rooms.append(rooms)
    
    # 建立三角形邻接关系
    adjacency = find_adjacent_triangles(triangles)
    
    # 标记特殊三角形
    is_special = [False] * len(triangles)
    
    for tri_idx in range(len(triangles)):
        if is_special[tri_idx]:  # 已标记为特殊三角形
            continue
            
        tri = triangles[tri_idx]
        
        # 检查当前三角形是否属于房间内部三角形
        if triangle_room_ids[tri_idx] != -1:
            continue  # 是房间内部三角形，跳过
            
        # 检查当前三角形所有顶点是否都在房间边上
        all_on_edge = True
        for v_idx in tri:
            if not vertex_rooms[v_idx]:  # 顶点不在任何房间边上
                all_on_edge = False
                break
        if not all_on_edge:
            continue
            
        # 找到最大内角的对边
        max_angle_edge = find_max_angle_opposite_edge(vertices, tri)
        
        # 找到共享这条边的相邻三角形
        adjacent = None
        for adj_idx, edge in adjacency[tri_idx]:
            if edge == max_angle_edge:
                adjacent = adj_idx
                break
                
        if adjacent is None:  # 没有相邻三角形
            continue
            
        # 检查相邻三角形是否属于房间内部三角形
        if triangle_room_ids[adjacent] != -1:
            continue  # 是房间内部三角形，跳过
            
        # 检查相邻三角形所有顶点是否都在房间边上
        adj_tri = triangles[adjacent]
        adj_all_on_edge = True
        for v_idx in adj_tri:
            if not vertex_rooms[v_idx]:  # 顶点不在任何房间边上
                adj_all_on_edge = False
                break
        if not adj_all_on_edge:
            continue
            
        # 满足所有条件，标记为特殊三角形
        is_special[tri_idx] = True
        is_special[adjacent] = True
    
    logging.info(f"[DEBUG] 识别到 {sum(is_special)} 个特殊三角形")
    
    # 可视化结果 - 创建多幅图
    fig = plt.figure(figsize=(25, 15))
    
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
    # 只有在重新三角化时才有wall_segments，否则需要重新生成
    for seg in wall_segments:
        x, y = zip(*seg)
        ax2.plot(x, y, 'g-', linewidth=1.5, label='延伸后墙面线段' if ax2.get_legend_handles_labels()[1] == [] else "")
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.set_title('2. 延伸至边界框的墙面线段')
    
    # 3. 三角剖分结果
    ax3 = fig.add_subplot(223)
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax3.triplot(triang, color='lightblue', linewidth=0.5)
    ax3.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)
    # 绘制边界框
    ax3.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.set_title('3. 约束Delaunay三角剖分结果')
    
    # 4. 按房间和特殊三角形着色的结果
    ax4 = fig.add_subplot(224)
    # 创建颜色映射（房间颜色 + 特殊三角形颜色 + 未分配区域颜色）
    cmap_colors = [room_color_map[rid] for rid in sorted(room_polygons.keys())]
    cmap_colors.append([1.0, 0.5, 0.0])  # 橙色 - 特殊三角形
    cmap_colors.append([0.8, 0.8, 0.8])  # 灰色 - 未分配区域
    
    # 调整ID以匹配颜色索引
    adjusted_ids = []
    for i in range(len(triangle_room_ids)):
        if is_special[i]:
            adjusted_ids.append(len(room_polygons))  # 特殊三角形使用橙色
        else:
            # 普通三角形使用房间颜色或灰色
            rid = triangle_room_ids[i]
            if rid == -1:
                adjusted_ids.append(len(room_polygons) + 1)
            else:
                adjusted_ids.append(sorted(room_polygons.keys()).index(rid))
    
    cmap = ListedColormap(cmap_colors)
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax4.tripcolor(triang, adjusted_ids, cmap=cmap, alpha=0.7)
    # 绘制三角形边界
    ax4.triplot(triang, color='k', linewidth=0.5)
    # 绘制边界框
    ax4.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    # 绘制墙面线段（如果存在）
    for seg in wall_segments:
        x, y = zip(*seg)
        ax4.plot(x, y, 'g-', linewidth=1.5, label='墙面线段' if ax4.get_legend_handles_labels()[1] == [] else "")
    
    # 添加图例
    handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in cmap_colors[:-2]]
    labels = [f'房间 {rid}' for rid in sorted(room_polygons.keys())]
    handles.append(plt.Rectangle((0,0),1,1, facecolor=cmap_colors[-2]))
    labels.append('特殊三角形')
    handles.append(plt.Rectangle((0,0),1,1, facecolor=cmap_colors[-1]))
    labels.append('墙体/外部区域')
    ax4.legend(handles, labels, loc='upper right')
    
    ax4.set_aspect('equal')
    ax4.set_title(f'4. 按房间和特殊三角形着色的结果（特殊三角形: {sum(is_special)}个）')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(output_image_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logging.info("[SUCCESS] 可视化结果已保存！")

    # =======================另存ply，携带room_id和内墙标记======================
    # 验证属性数据有效性
    if len(triangle_room_ids) != triangles.shape[0]:
        logging.error(f"[ERROR] triangle_room_ids 长度({len(triangle_room_ids)})与三角形数({triangles.shape[0]})不匹配，跳过PLY保存")
        return
    
    if len(is_special) != triangles.shape[0]:
        logging.error(f"[ERROR] is_special 长度({len(is_special)})与三角形数({triangles.shape[0]})不匹配，跳过PLY保存")
        return
    
    json_filename = os.path.basename(json_path)
    file_name = os.path.join('DelaunayTriangleMesh_With_RoomClass/sample0_256', json_filename.replace('.json', '.ply'))
    print(f'另存ply {file_name}')

    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)
    save_ply_with_face_attrs(
            vertices=vertices,
            triangles=triangles,
            triangle_room_ids=triangle_room_ids,
            is_special=is_special,
            filename=file_name,
            binary=True
        )

    # =======================提取封闭区域外边界======================
    # 新增：提取封闭区域外边界
    logging.info("[DEBUG] 开始提取封闭区域外边界...")
    enclosed_regions = extract_enclosed_regions(vertices, triangles, triangle_room_ids, is_special)
    logging.info(f"[DEBUG] 提取到 {len(enclosed_regions)} 个封闭区域")
    
    # 新增：可视化封闭区域外边界
    fig_extra = plt.figure(figsize=(12, 12))
    ax5 = fig_extra.add_subplot(111)
    
    # 绘制所有三角形
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax5.tripcolor(triang, adjusted_ids, cmap=cmap, alpha=0.5)
    
    # 绘制封闭区域外边界
    colors = plt.cm.Set3(np.linspace(0, 1, len(enclosed_regions)))
    for i, region in enumerate(enclosed_regions):
        if len(region) < 3:
            continue  # 忽略点数不足的多边形
        x, y = zip(*region)
        x += (x[0],)  # 闭合多边形
        y += (y[0],)
        ax5.plot(x, y, '-', color=colors[i], linewidth=3, label=f'封闭区域 {i+1}')
    
    # 绘制边界框
    ax5.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')
    
    ax5.set_aspect('equal')
    ax5.legend()
    ax5.set_title(f'封闭区域外边界（共 {len(enclosed_regions)} 个）')
    
    # 保存封闭区域可视化结果
    region_image_name = output_image_name.replace('.png', '_regions.png')
    plt.tight_layout()
    plt.show()
    # plt.savefig(region_image_name, dpi=300, bbox_inches='tight')
    plt.close(fig_extra)
    
    # # 可以将封闭区域结果保存为JSON
    # regions_data = {
    #     'enclosed_regions': enclosed_regions,
    #     'region_count': len(enclosed_regions)
    # }
    # regions_json_name = output_image_name.replace('.png', '_regions.json')
    # with open(regions_json_name, 'w', encoding='utf-8') as f:
    #     json.dump(regions_data, f, ensure_ascii=False, indent=2)
    
    logging.info("[SUCCESS] 封闭区域提取完成并保存！")


if __name__ == '__main__':
    json_path = '../SpatiallmDataProcessor/output/coco_with_scaled/sample0_256/anno/scene_000000.json'
    # ply_path = 'triangle_triangulation.ply'
    ply_path = 'cgal_triangulation.ply'
    output_image_name = os.path.join('./', os.path.basename(json_path).replace('.json', '.png'))
    process_and_visualize_scene(json_path, output_image_name, ply_path)
    # 假设所有JSON文件和PLY文件都在这些目录下
    # resume_from_idx = 0

    # anno_dir = '../SpatiallmDataProcessor/output/coco_with_scaled/sample0_256/anno/'
    # ply_dir = 'DelaunayTriangleMesh/sample0_256'  # PLY文件保存目录
    # os.makedirs(ply_dir, exist_ok=True)  # 确保目录存在

    # json_files = sorted(glob.glob(os.path.join(anno_dir, 'scene_*.json')))  # 按名字排序
    # for idx, json_path in enumerate(json_files[resume_from_idx:], start=resume_from_idx):
    #     json_filename = os.path.basename(json_path)         # 如 'scene_000000.json'
    #     ply_filename = json_filename.replace('.json', '.ply')  # 如 'scene_000000.ply'
    #     ply_path = os.path.join(ply_dir, ply_filename)       # 如 'DelaunayTriangleMesh/sample0_256/scene_000000.ply'
    #     output_image_name = os.path.join('rule_output/rule_test', json_filename.replace('.json', '.png'))
    #     print(f"正在处理第 {idx+1} 个场景: {json_path}")
    #     logging.info(f"正在处理第 {idx+1} 个场景: {json_path}")
    #     try:
    #         process_and_visualize_scene(json_path, output_image_name, ply_path)
    #     except Exception as e:
    #         print(f"[ERROR] 处理 {json_path} 时出错: {e}")
    #         logging.error(f"[ERROR] 处理 {json_path} 时出错: {e}")

