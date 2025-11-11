import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# 1. 读取 JSON 文件
json_path = '00002.json' # ← 请替换为你的 JSON 文件路径
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 加载图像（假设是 00000.png，与 JSON 同目录）
image_id = 2
image_info = next((img for img in data['images'] if img['id'] == image_id), None)
if not image_info:
    raise ValueError("未找到 id=0 的图像")

image_filename = image_info['file_name']
image_path = image_filename  # 如果图像不在当前目录，请填写完整路径，如：'path/to/00000.png'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
height, width = image.shape[:2]

# 3. 类别信息
category_info = {cat['id']: cat for cat in data['categories']}
# 定义颜色（仅示例，你可以根据需要扩展）
colors = {
    0: 'lightblue',    # living room
    2: 'lightgreen',   # bedroom
    3: 'lightcoral',   # bathroom
    6: 'gold',         # dining room
    16: 'red',         # door
    17: 'cyan',        # window
}
default_color = 'orange'  # 未知类别默认颜色

# 4. 绘图
fig, ax = plt.subplots(1, figsize=(12, 12))

# 显示图像
ax.imshow(image)

# 可选择性显示坐标轴刻度（默认就有，但可以调大字体方便看）
ax.set_xlabel('X (pixel)', fontsize=12)
ax.set_ylabel('Y (pixel)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.grid(True, linestyle='--', alpha=0.6)  # 可选：显示网格

# 5. 遍历所有 annotations
for ann in data['annotations']:
    category_id = ann['category_id']
    cat_name = category_info.get(category_id, {}).get('name', f'unknown_{category_id}')
    color = colors.get(category_id, default_color)

    # --- 绘制 segmentation（多边形）---
    if 'segmentation' in ann and ann['segmentation']:
        for seg in ann['segmentation']:
            poly_pts = np.array(seg).reshape(-1, 2).astype(int)
            if len(poly_pts) >= 3:
                poly = Polygon(poly_pts, closed=True, fill=True, 
                              edgecolor='black', facecolor=color, alpha=0.4)
                ax.add_patch(poly)
                # 可选：在多边形中心显示类别名称和坐标
                centroid = np.mean(poly_pts, axis=0)
                ax.text(centroid[0], centroid[1], f'{cat_name}\n({centroid[0]:.0f}, {centroid[1]:.0f})',
                        fontsize=7, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # --- 绘制 bbox（边界框）---
    if 'bbox' in ann and ann['bbox']:
        x, y, w, h = map(int, ann['bbox'])
        rect = Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        # 在 bbox 左上角显示类别和坐标
        ax.text(x, y - 5, f'{cat_name}\n({x},{y})', fontsize=8, color=color,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# 6. 设置坐标轴范围匹配图像尺寸
ax.set_xlim(0, width)
ax.set_ylim(height, 0)  # 注意：matplotlib 的 y 轴是向下的，所以要 height -> 0

# 可选：显示像素网格或调整坐标轴密度
ax.set_xticks(np.arange(0, width, 20))
ax.set_yticks(np.arange(0, height, 20))
plt.grid(True, linestyle=':', alpha=0.5)

# 7. 显示结果
plt.title(f'Image: {image_filename} | 带坐标轴的标注可视化 (X/Y pixel)', fontsize=14)
plt.tight_layout()
plt.show()

# 如需保存带坐标轴的图像，可取消注释以下代码：
# output_filename = 'annotated_with_axes.png'
# fig.savefig(output_filename, dpi=300, bbox_inches='tight')
# print(f" 已保存带坐标轴的可视化图像到：{output_filename}")