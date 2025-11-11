import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle  # pip install triangle

def save_triangulation_to_ply(vertices, triangles, filename='output_mesh.ply', binary=True):
    """
    将三角剖分结果保存为 PLY 文件格式
    
    Args:
        vertices: (N, 2) 或 (N, 3) 的 numpy 数组，表示顶点坐标，比如 [[x1,y1], [x2,y2], ...]
        triangles: (M, 3) 的 numpy 数组，每个元素是三个顶点的索引，比如 [[i1,i2,i3], ...]
        filename: 输出的 PLY 文件名，比如 'mesh.ply'
        binary: 是否保存为二进制 PLY（推荐 True，文件更小更快）
    """
    import os

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
            # 按照 x, y, z 的顺序写入每个顶点为 float32
            vertices_flat = vertices.astype(np.float32).flatten()
            f.write(vertices_flat.tobytes())
        else:
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # 写入 faces（每个面是 3 个顶点索引）
        if binary:
            for face in triangles:
                # PLY binary 格式中，face 的开头是一个 uchar（1字节）表示顶点数，然后是 int 索引
                face_header = np.array([3], dtype=np.uint8)  # 三角形 => 3 个顶点
                face_indices = face.astype(np.int32)  # 顶点索引
                f.write(face_header.tobytes())
                f.write(face_indices.tobytes())
        else:
            for face in triangles:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"[INFO] 三角网格已保存为 PLY 文件: {os.path.abspath(filename)}")

# =============================================
# 1. 模拟从 S3D.json 加载的 annotations 数据
# （这里用您提供的示例格式，您可以替换为真实读取 json 的代码）
import json

# 从文件加载 S3D.json
with open('s3d/00000.json', 'r', encoding='utf-8') as f:
    s3d_data = json.load(f)

annotations = s3d_data.get('annotations', [])
# 排除 id 为 16 和 17 的 annotation（门/窗）
annotations = [ann for ann in annotations if ann['category_id'] not in [16, 17]]

# =============================================
# 2. 定义外包围盒（bounding box），可以自定义或从数据中计算
# 比如：整个场景范围，或者您希望的一个大矩形
# 这里我们先自动计算所有顶点的最小/最大边界，然后扩大一点作为包围盒
all_points = []

for ann in annotations:
    seg = ann["segmentation"][0]  # coco格式: [x1,y1,x2,y2,...]
    points = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
    all_points.extend(points)

if not all_points:
    raise ValueError("没有找到任何有效的房间多边形顶点！")

""" # 计算包围盒
xs = [p[0] for p in all_points]
ys = [p[1] for p in all_points]
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

# 外包围盒：稍微放大一点，确保包含所有房间
padding = 10
box = [
    (min_x - padding, min_y - padding),
    (max_x + padding, min_y - padding),
    (max_x + padding, max_y + padding),
    (min_x - padding, max_y + padding)
] """

box = [
    (0, 0),
    (256, 0),
    (256, 256),
    (0, 256)
]
# =============================================
# 3. 转换所有房间 segmentation 为多边形点列表
rooms = []
for ann in annotations:
    seg = ann["segmentation"][0]  # coco格式: [x1,y1,x2,y2,...]
    polygon = [(seg[i * 2], seg[i * 2 + 1]) for i in range(len(seg) // 2)]
    rooms.append(polygon)

# =============================================
# 4. 构造 triangle 所需的 vertices 和 segments

# 所有点：先放外包围盒，然后各个房间
points = box.copy()
offset_box = len(box)

for room in rooms:
    points.extend(room)
offset_room_start = offset_box

# 构建 segments（边，必须是闭合的环）
def polygon_edges(offset, polygon):
    """生成一个多边形的边，每个边是 (点i, 点i+1)，最后闭合"""
    edges = []
    n = len(polygon)
    for i in range(n):
        edges.append((offset + i, offset + (i + 1) % n))
    return edges

segments = []
# 外包围盒的边
segments += polygon_edges(0, box)

# 各个房间的边
for i, room in enumerate(rooms):
    offset = offset_box + sum(len(r) for r in rooms[:i])
    segments += polygon_edges(offset, room)

print("[DEBUG] Number of vertices:", len(points))
print("[DEBUG] vertices.shape:", np.array(points).shape)  # 必须是 (N, 2)

print("[DEBUG] Number of segments:", len(segments))
print("[DEBUG] First 5 segments:", segments[:5])  # 打印前5条边看看是否合理

# 强烈建议：确保 segments 是整数，且是二维的 (N, 2)
segments_np = np.array(segments, dtype=int)
print("[DEBUG] segments_np.shape:", segments_np.shape)  # 必须是 (N, 2)
print("[DEBUG] segments_np.dtype:", segments_np.dtype)  # 必须是 int

# 检查所有 segment 的索引是否没有越界
max_idx = segments_np.max()
if max_idx >= len(points):
    raise ValueError(f"❌ Segment index {max_idx} 超出了 vertices 的范围（总点数={len(points)}）。请检查 segments 的生成。")

# 构造输入字典
A = dict(
    vertices=np.array(points, dtype=float),  # 必须是 float64 类型的 Nx2 数组
    segments=segments_np  # 必须是 int 类型的 Nx2 数组
)

print("[DEBUG] Calling triangle.triangulate...")
B = triangle.triangulate(A, 'p')
print("[DEBUG] Triangulation succeeded!")



if 'triangles' not in B:
    print("❌ 三角剖分失败，未生成任何三角形。请检查输入的 segments 是否正确闭合。")
else:
    # 7. ===============================可视化结果===============================
    vertices = B['vertices'] # shape: (N, 2)
    triangles = B['triangles'] # shape: (M, 3)
    # # ✅ 翻转 Y 轴，解决可视化时的上下颠倒问题
    # vertices[:, 1] = -vertices[:, 1]


    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    plt.figure(figsize=(10, 10))
    plt.triplot(triang, color='lightblue', linewidth=0.5)
    plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='red', markersize=2)

    # 可选：把外框和房间边界再画一遍（更清晰）
    # 外包围盒
    box_x, box_y = zip(*box)
    plt.plot(box_x + (box_x[0],), box_y + (box_y[0],), 'r-', linewidth=2, label='Bounding Box')

    # 房间边界
    color_idx = 0
    colors = ['green', 'orange', 'purple', 'brown']
    for i, room in enumerate(rooms):
        room_x, room_y = zip(*room)
        plt.plot(room_x + (room_x[0],), room_y + (room_y[0],), '--', color=colors[color_idx % len(colors)], linewidth=1.5, label=f'Room {i+1}' if i < len(colors) else "")
        color_idx += 1

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Constrained Delaunay Triangulation (CDT) with Rooms & Bounding Box')
    plt.show()

    # ===============================保存ply===============================
    # 保存为 PLY（二进制，推荐）
    save_triangulation_to_ply(vertices, triangles, filename='triangle_mesh_output.ply', binary=True)

    print("[SUCCESS] 三角剖分结果已保存为 PLY 文件！")