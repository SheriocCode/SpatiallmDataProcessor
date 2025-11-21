import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import json
import os

def save_triangulation_to_ply(vertices, triangles, filename='output_mesh.ply', binary=True):
    """将三角剖分结果保存为 PLY 文件格式"""
    # 确保 vertices 是 N x 3，如果不是，补一个 z=0 列
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
        # 写入 header
        if binary:
            f.write(header.encode('ascii'))
        else:
            f.write(header)

        # 写入 vertices
        if binary:
            vertices_flat = vertices.astype(np.float32).flatten()
            f.write(vertices_flat.tobytes())
        else:
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # 写入 faces
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

# 将线段延伸至边界框
def extend_segment_to_bbox(segment, bbox):
    (x1, y1), (x2, y2) = segment
    box_min_x, box_min_y = bbox[0]
    box_max_x, box_max_y = bbox[2]

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0: # 垂直线
        x = x1
        return [(x, box_min_y), (x, box_max_y)]
    if dy == 0: # 水平线
        y = y1
        return [(box_min_x, y), (box_max_x, y)]

    # 计算与边界框的交点
    intersections = []
    m = dy / dx  # 斜率
    b = y1 - m * x1  # 截距
    # 计算与左右边界的交点
    y_left = m * box_min_x + b
    y_right = m * box_max_x + b
    # 计算与上下边界的交点
    x_bottom = (box_min_y - b) / m if m != 0 else None
    x_top = (box_max_y - b) / m if m != 0 else None

    # 筛选有效的边界交点
    if box_min_y <= y_left <= box_max_y:
        intersections.append((box_min_x, y_left))
    if box_min_y <= y_right <= box_max_y:
        intersections.append((box_max_x, y_right))
    if box_min_x <= x_bottom <= box_max_x:
        intersections.append((x_bottom, box_min_y))
    if box_min_x <= x_top <= box_max_x:
        intersections.append((x_top, box_max_y))

    return intersections

def get_vertices_and_segments(annotations, box):
    # 提取所有墙面线段并延伸至边界框
    wall_segments = []
    all_points = []

    for ann in annotations:
        seg = ann["segmentation"][0]  # coco格式: [x1, y1, x2, y2, ..., xn, yn]
        polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
        # 提取多边形的边作为墙面线段
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i] # 起点 (xi, yi)
            p2 = polygon[(i + 1) % n] # 终点 (x(i+1)%n, y(i+1)%n)
            # 延伸线段至边界框
            extended = extend_segment_to_bbox((p1, p2), box)
            wall_segments.append(extended)
            all_points.extend(extended)

    # print("[Debug] wall_segments")
    # print(wall_segments)
    # print("[Debug] all_points")
    # print(all_points)

    # 添加边界框的点
    all_points.extend(box)
    # 去重顶点
    unique_points = []
    seen = set()
    for p in all_points:
        # 用四舍五入解决浮点数精度问题
        # key = (round(p[0], 6), round(p[1], 6))
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique_points.append(p)

    # 顶点索引
    # point_indices = { (round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(unique_points) }
    point_indices = { (p[0], p[1]): i for i, p in enumerate(unique_points) }

    # 去重线段
    unique_segments = []
    seen_seg = set()
    for seg in wall_segments:
        # 标准化线段表示（按点索引排序，避免(a,b)和(b,a)被视为不同）
        p1, p2 = seg
        # key = tuple(sorted([
        #     (round(p1[0], 6), round(p1[1], 6)),
        #     (round(p2[0], 6), round(p2[1], 6))
        # ]))
        key = tuple(sorted([
            (p1[0], p1[1]),
            (p2[0], p2[1])
        ]))
        if key not in seen_seg:
            seen_seg.add(key)
            unique_segments.append(seg)

    segments = []
    for seg in unique_segments:
        p1, p2 = seg
        # idx1 = point_indices[(round(p1[0], 6), round(p1[1], 6))]
        # idx2 = point_indices[(round(p2[0], 6), round(p2[1], 6))]
        idx1 = point_indices[(p1[0], p1[1])]
        idx2 = point_indices[(p2[0], p2[1])]
        segments.append((idx1, idx2))

    # 添加边界框的边
    # box_indices = [point_indices[(round(p[0], 6), round(p[1], 6))] for p in box]
    box_indices = [point_indices[(p[0], p[1])] for p in box]
    n = len(box_indices)
    for i in range(n):
        segments.append((box_indices[i], box_indices[(i + 1) % n]))

    # print("[DEBUG] 顶点数量:", len(unique_points))
    # print("[DEBUG] 线段数量:", len(segments))

    # 检查索引是否越界
    # segments_np = np.array(segments, dtype=int)
    # max_idx = segments_np.max()
    # if max_idx >= len(unique_points):
    #     raise ValueError(f"❌ 线段索引 {max_idx} 超出了顶点范围（总顶点数={len(unique_points)}）")
    
    return unique_points, segments

def cgal_constrained_triangulation(vertices, segments):
    """使用CGAL进行约束Delaunay三角剖分（带可视化）"""
    from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
    from CGAL.CGAL_Kernel import Point_2

    # 1. 转换顶点为CGAL的Point_2类型
    cgal_points = [Point_2(v[0], v[1]) for v in vertices]
    
    # 2. 初始化约束Delaunay三角剖分
    cdt = Constrained_Delaunay_triangulation_2()
    
    # 3. 插入所有原始顶点并记录句柄
    vertex_handles = [cdt.insert(p) for p in cgal_points]
    
    # 4. 插入约束线段（使用原始顶点句柄）
    for seg in segments:
        idx1, idx2 = seg
        vh1 = vertex_handles[idx1]
        vh2 = vertex_handles[idx2]
        cdt.insert_constraint(vh1, vh2)
    
    # 5. 收集所有顶点（原始顶点 + 剖分新增顶点）
    all_vertices = []
    handle_to_index = {}  # 句柄到索引的映射
    # 遍历所有有限顶点
    for vh in cdt.finite_vertices():
        if vh not in handle_to_index:
            handle_to_index[vh] = len(all_vertices)
            p = vh.point()
            all_vertices.append([p.x(), p.y()])  # 转换为Python列表
    
    # 6. 提取三角形（映射到新的顶点索引）
    triangles = []
    for face in cdt.finite_faces():
        # 跳过无限面（边界外的面）
        if cdt.is_infinite(face):
            continue
        # 获取三个顶点的句柄
        vh0 = face.vertex(0)
        vh1 = face.vertex(1)
        vh2 = face.vertex(2)
        # 查找对应的索引
        idx0 = handle_to_index[vh0]
        idx1 = handle_to_index[vh1]
        idx2 = handle_to_index[vh2]
        triangles.append([idx0, idx1, idx2])
    
    # 转换为numpy数组
    vertices_np = np.array(all_vertices, dtype=float)
    triangles_np = np.array(triangles, dtype=int)

    # 可视化结果
    if len(triangles_np) > 0:
        return vertices_np, triangles_np
    else:
        print("❌ CGAL三角剖分失败，未生成任何三角形")
        return None, None

def triangle_constrained_triangulation(vertices, segments):
    import triangle
    """使用triangle进行三角剖分"""
    # 构造输入字典
    A = dict(
        vertices=np.array(vertices, dtype=float),
        segments=segments
    )

    print("[DEBUG] 开始三角剖分...")
    B = triangle.triangulate(A, 'p')
    print("[DEBUG] 三角剖分完成!")

    if 'triangles' not in B:
        print("❌ 三角剖分失败，未生成任何三角形")
    else:
        return B['vertices'], B['triangles']

def visualize_plt(vertices, triangles):
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    plt.figure(figsize=(10, 10))
    plt.triplot(triang, color='lightblue', linewidth=0.5)
    plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)

    # 绘制边界框
    box_x, box_y = zip(*box)
    plt.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')

    # 绘制延伸后的墙面线段
    for seg in wall_segments:
        idx1, idx2 = seg  # seg 是一个索引对，比如 (3, 5)
        p1 = unique_points[idx1]  # 从 unique_points 中取出第 idx1 个点（坐标）
        p2 = unique_points[idx2]  # 取出第 idx2 个点
        x, y = [p1[0], p2[0]], [p1[1], p2[1]]  # 提取 x 和 y 坐标
        plt.plot(x, y, 'g-', linewidth=1.5)  # 画线

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('延伸墙面线段后的约束Delaunay三角剖分')
    plt.show()

# 定义外包围盒
box = [
    (0, 0),
    (256, 0),
    (256, 256),
    (0, 256)
]
# 加载 coco_with_scaled 数据
with open('../SpatiallmDataProcessor/output/coco_with_scaled/sample0_256/anno/scene_004512.json', 'r', encoding='utf-8') as f:
# with open('../SpatiallmDataProcessor/output/coco_with_scaled/sample0_256/anno/scene_000000.json', 'r', encoding='utf-8') as f:
    coco_data = json.load(f)
annotations = coco_data.get('annotations', [])
# 排除 id 为 0 和 1 的 annotation（窗/门）
annotations = [ann for ann in annotations if ann['category_id'] not in [0, 1]]

unique_points, wall_segments = get_vertices_and_segments(annotations, box)

# CGAL三角剖分
print("[DEBUG] 开始三角剖分（CGAL）...")
# vertices_np = np.array(unique_points, dtype=float)
vertices, triangles = cgal_constrained_triangulation(unique_points, wall_segments)
print("[DEBUG] 三角剖分（CGAL）完成!")
visualize_plt(vertices, triangles)

# Triangle 三角剖分
print("[DEBUG] 开始三角剖分（Triangle）...")
vertices, triangles = triangle_constrained_triangulation(unique_points, wall_segments)
print("[DEBUG] 三角剖分（Triangle）完成!")
visualize_plt(vertices, triangles)

