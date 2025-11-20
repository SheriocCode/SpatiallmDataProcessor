import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import triangle  # pip install triangle

# -----------------------------
# 定义外包围盒 polygon（必须闭合）
box = [(0, 0), (256, 0), (256, 256), (0, 256)]

# 假设两个房间，都是凸多边形（顺时针或逆时针都行）
room1 = [(50, 50), (100, 50), (100, 100), (50, 100)]
room2 = [(150, 150), (200, 150), (200, 200), (150, 200)]

# -----------------------------
# 收集所有点 + 构造 segments（边）
points = box + room1 + room2

def polygon_edges(offset, polygon):
    """给定一个多边形点列表和偏移量，生成多边形的边（点索引对）"""
    return [(i + offset, (i + 1) % len(polygon) + offset) for i in range(len(polygon))]

segments = []
segments += polygon_edges(0, box)
segments += polygon_edges(len(box), room1)
segments += polygon_edges(len(box) + len(room1), room2)

# 构造 Triangle 格式输入
A = dict(
    vertices=np.array(points),
    segments=np.array(segments)
)

# -----------------------------
# 进行 CDT 三角剖分
B = triangle.triangulate(A, 'p')  # 'p' = 保留 segments

# -----------------------------
# 可视化结果
if 'triangles' not in B:
    print("三角剖分失败，没有生成任何三角形。")
    exit()

vertices = B['vertices']
triangles = B['triangles']

triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

plt.figure(figsize=(8, 8))
plt.triplot(triang, color='gray')
plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='blue', markersize=2)
plt.gca().set_aspect('equal')
plt.title('Constrained Delaunay Triangulation (with outer box and rooms)')
plt.show()
