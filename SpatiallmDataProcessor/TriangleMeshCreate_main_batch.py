# 添加超时处理（使用超时库）->无效
# 使用 multiprocessing 处理超时

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle  # pip install triangle
import json
import os
import logging
from utils.log_util import init_logger
import timeout_decorator

init_logger('DelaunayTriangleMesh/.log')

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

    # 定义边界框的四条边
    bbox_edges = [
        [(min_x, min_y), (max_x, min_y)],  # 底边
        [(max_x, min_y), (max_x, max_y)],  # 右边
        [(max_x, max_y), (min_x, max_y)],  # 顶边
        [(min_x, max_y), (min_x, min_y)]   # 左边
    ]

    # 计算线段的参数方程
    dx = x2 - x1
    dy = y2 - y1

    # 如果是垂直线
    if abs(dx) <= 2: 
    # if dx == 0:
        x = x1
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        return [(x, min_y), (x, max_y)]
    
    # 如果是水平线
    if abs(dy) <= 2: 
    # if dy == 0:
        y = y1
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        return [(min_x, y), (max_x, y)]

    # 计算与边界框的交点
    intersections = []
    for edge in bbox_edges:
        intersect = line_intersection(segment, edge)
        if intersect:
            intersections.append(intersect)

    # 如果没有交点（线段完全在边界框内），延长至边界
    if len(intersections) < 2:
        m = dy / dx  # 斜率
        b = y1 - m * x1  # 截距

        # 计算与左右边界的交点
        y_left = m * min_x + b
        y_right = m * max_x + b

        # 计算与上下边界的交点
        x_bottom = (min_y - b) / m if m != 0 else None
        x_top = (max_y - b) / m if m != 0 else None

        # 筛选有效的边界交点
        if min_y <= y_left <= max_y:
            intersections.append((min_x, y_left))
        if min_y <= y_right <= max_y:
            intersections.append((max_x, y_right))
        if x_bottom is not None and min_x <= x_bottom <= max_x:
            intersections.append((x_bottom, min_y))
        if x_top is not None and min_x <= x_top <= max_x:
            intersections.append((x_top, max_y))

    # 取两个最远的点作为延伸后的线段端点
    if len(intersections) >= 2:
        dists = [((x - x1)**2 + (y - y1)** 2) for x, y in intersections]
        far1_idx = np.argmax(dists)
        far1 = intersections.pop(far1_idx)
        
        dists = [((x - far1[0])**2 + (y - far1[1])** 2) for x, y in intersections]
        far2_idx = np.argmax(dists)
        far2 = intersections[far2_idx]
        
        return [far1, far2]
    
    return segment  # 无法延伸的情况


# 抽取函数：可视化延伸墙面线段后的约束Delaunay三角剖分
def visualize_plt(vertices, triangles, wall_segments, box):
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    plt.figure(figsize=(10, 10))
    plt.triplot(triang, color='lightblue', linewidth=0.5)
    plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)

    # 绘制边界框
    box_x, box_y = zip(*box)
    plt.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='边界框')

    # 绘制延伸后的墙面线段
    for seg in wall_segments:
        x, y = zip(*seg)
        plt.plot(x, y, 'g-', linewidth=1.5)

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('延伸墙面线段后的约束Delaunay三角剖分')
    plt.show()

# @timeout_decorator.timeout(5, timeout_exception=TimeoutError)
# def safe_triangulate(A):
#     """带超时保护的三角剖分函数"""
#     return triangle.triangulate(A, 'p')  # 使用前面的参数

def triangulate_task(A, result_queue):
    try:
        B = triangle.triangulate(A, 'p')
        result_queue.put(('success', B))
    except Exception as e:
        result_queue.put(('error', str(e)))

def safe_triangulate(A, timeout=4):
    """
    带超时控制的三角剖分函数
    :param A: 三角剖分输入数据
    :param timeout: 超时时间（秒）
    :return: 成功返回结果，超时/错误返回None
    """
    # 创建队列用于接收子进程结果
    result_queue = multiprocessing.Queue()
    # 创建子进程
    p = multiprocessing.Process(
        target=triangulate_task,
        args=(A, result_queue)
    )
    # 启动子进程
    p.start()
    # 等待超时时间（最多timeout秒）
    p.join(timeout)
    
    # 判断子进程状态
    if p.is_alive():
        # 超时：强制终止子进程
        p.terminate()
        p.join()  # 确保进程彻底终止
        return ('timeout', None)
    else:
        # 未超时：获取结果
        if not result_queue.empty():
            return result_queue.get()
        else:
            return ('error', '未知错误')

def process_result(B, ply_save_path):
    vertices = B['vertices']
    triangles = B['triangles']
    # 可视化结果
    # visualize_plt(vertices, triangles, wall_segments, box)
    # 保存为PLY文件
    save_triangulation_to_ply(vertices, triangles, filename=ply_save_path, binary=True)

def generate_Delaunay_Mesh(coco_file_path, output_dir):

    # 加载 coco_with_scaled 数据
    output_path = os.path.join(output_dir,  os.path.basename(coco_file_path).replace('.json', '.ply'))
    with open(coco_file_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    annotations = coco_data.get('annotations', [])
    annotations = [ann for ann in annotations if ann['category_id'] not in [0, 1]]


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
    all_points = []

    for ann in annotations:
        seg = ann["segmentation"][0]  # coco格式: [x1,y1,x2,y2,...]
        polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
        
        # 提取多边形的边作为墙面线段
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
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

    # print("[DEBUG] 顶点数量:", len(unique_points))
    # print("[DEBUG] 线段数量:", len(segments))

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

    result_type, result = safe_triangulate(A, timeout=4)  # 超时时间4秒
    if result_type == 'success':
        # 处理成功，继续后续操作
        B = result
        process_result(B, output_path)
    elif result_type == 'timeout':
        # 超时（卡死），记录并跳过
        print(f"【超时错误】文件 {coco_file_path} 处理超时")
        logging.error(f"【超时错误】文件 {coco_file_path} 处理超时")

    else:
        # 其他错误，记录并跳过
        logging.error(f"【错误】文件 {coco_file_path} 三角剖分失败")


if __name__ == '__main__':
    OUTPUT_DIR = 'DelaunayTriangleMesh/'
    COCO_WITH_SCALED_FOLDER = 'coco_with_scaled'
    for sample_dir in os.listdir(COCO_WITH_SCALED_FOLDER):
        if sample_dir.startswith('sample'):
            # e.g. coco_with_scaled/sample0_256/anno
            anno_dir = os.path.join(COCO_WITH_SCALED_FOLDER, sample_dir, 'anno')
            
            # 遍历 anno 目录下的所有 json 文件
            for json_file in os.listdir(anno_dir):
                if json_file.endswith('.json'):
                    # 构造完整的 json 文件路径
                    json_path = os.path.join(anno_dir, json_file)
                    # 构造输出路径
                    output_path = os.path.join(OUTPUT_DIR, sample_dir)
                    os.makedirs(output_path, exist_ok=True)
                    # 生成 Delaunay 三角剖分
                    generate_Delaunay_Mesh(json_path, output_path)
