import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

def visualize_coco_on_density_with_lines(coco_json_path, density_image_path, 
                                         output_path="visualized_density.png", 
                                         alpha=0.4, line_thickness=2):
    """
    将COCO标注可视化到点云密度图上（支持线段标注）
    
    参数:
        coco_json_path: COCO格式JSON文件路径
        density_image_path: 点云密度图路径
        output_path: 输出图像路径
        alpha: 分割区域透明度 (0-1)
        line_thickness: 边界框和线段宽度
    """
    
    # 1. 读取数据
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 读取点云密度图
    image = cv2.imread(density_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {density_image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    
    # 3. 生成颜色映射
    category_ids = set(ann['category_id'] for ann in coco_data['annotations'])
    color_map = {cat_id: np.random.rand(3,) for cat_id in category_ids}
    
    # 4. 按room_id分组
    annotations_by_room = {}
    for ann in coco_data['annotations']:
        room_id = ann.get('room_id', 0)
        if room_id not in annotations_by_room:
            annotations_by_room[room_id] = []
        annotations_by_room[room_id].append(ann)
    
    # 5. 绘制每个标注
    for room_id, anns in annotations_by_room.items():
        # 为每个room选择颜色变化
        room_color_shift = random.uniform(0.3, 1.0)
        
        for ann in anns:
            cat_id = ann['category_id']
            base_color = color_map[cat_id] * room_color_shift
            
            # 判断是否为线段标注（door/window等）
            is_line_annotation = (
                ann.get('area', 0) == 0.0 or 
                (len(ann.get('segmentation', [[]])[0]) < 6) or
                ann.get('bbox', [0,0,0,0])[3] == 0
            )
            
            if is_line_annotation:
                # --- 绘制线段标注（Door/Window）---
                seg = ann['segmentation'][0]
                if len(seg) >= 4:  # 至少需要2个点 (4个坐标)
                    x1, y1, x2, y2 = seg[:4]
                    
                    # 绘制线段，带端点标记
                    ax.plot([x1, x2], [y1, y2], 
                            color=base_color, 
                            linewidth=line_thickness + 2,  # 线段更粗更显眼
                            marker='o', 
                            markersize=5,
                            markerfacecolor='white',
                            markeredgecolor=base_color,
                            markeredgewidth=1.5)
                    
                    # 在线段中点添加标签
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    label_text = f"Room{room_id}-Cat{cat_id}"
                    if cat_id in [0, 1]:  # 根据实际类别ID调整
                        label_text += "(Door/Window)"
                    
                    ax.text(mid_x, mid_y - 8, label_text, 
                            color='white', 
                            fontsize=8, 
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round', 
                                    facecolor=base_color, 
                                    alpha=0.85,
                                    edgecolor='white',
                                    linewidth=0.5))
            
            else:
                # --- 绘制正常多边形标注（房间）---
                for polygon_coords in ann['segmentation']:
                    if len(polygon_coords) < 6:  # 至少需要3个点
                        continue
                    
                    # 转换为numpy数组 [N, 2]
                    poly = np.array(polygon_coords).reshape(-1, 2)
                    
                    # 过滤无效多边形
                    if len(poly) < 3 or calculate_area(poly) < 1.0:
                        continue
                    
                    # 绘制填充多边形
                    patch = Polygon(poly, closed=True, 
                                   facecolor=base_color, 
                                   alpha=alpha, 
                                   edgecolor='white', 
                                   linewidth=1)
                    ax.add_patch(patch)
                
                # --- 绘制边界框 ---
                bbox = ann['bbox']
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                    x, y, w, h = bbox
                    rect = plt.Rectangle((x, y), w, h, 
                                       linewidth=line_thickness,
                                       edgecolor=base_color, 
                                       facecolor='none',
                                       linestyle='--')  # 虚线框
                    ax.add_patch(rect)
                    
                    # 添加类别标签
                    label_text = f"Room{room_id}-Cat{cat_id}"
                    ax.text(x, y - 5, label_text, 
                            color='white', 
                            fontsize=8,
                            ha='left',
                            bbox=dict(boxstyle='round', 
                                    facecolor=base_color, 
                                    alpha=0.85,
                                    edgecolor='white',
                                    linewidth=0.5))
    
    # 6. 创建图例
    legend_elements = []
    for cat_id, color in color_map.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                           facecolor=color, 
                                           alpha=alpha,
                                           label=f"Category {cat_id}"))
    
    # 添加线段图例
    legend_elements.append(plt.Line2D([0], [0], 
                                    color='gray', 
                                    linewidth=line_thickness+2,
                                    marker='o', 
                                    markersize=5,
                                    label='Door/Window (Line)'))
    
    ax.legend(handles=legend_elements, 
              loc='upper right', 
              fontsize=8,
              framealpha=0.9)
    
    # 7. 保存和显示
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化结果已保存到: {output_path}")

def calculate_area(poly):
    """计算多边形面积"""
    if len(poly) < 3:
        return 0
    return cv2.contourArea(poly.astype(np.int32))

def check_and_count_annotations(coco_json_path):
    """检查并统计线段标注数量"""
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    total = len(data['annotations'])
    line_anns = 0
    normal_anns = 0
    
    for ann in data['annotations']:
        if (ann.get('area', 0) == 0 or 
            len(ann.get('segmentation', [[]])[0]) < 6 or
            ann.get('bbox', [0,0,0,0])[3] == 0):
            line_anns += 1
        else:
            normal_anns += 1
    
    print(f"标注统计:")
    print(f"  总数量: {total}")
    print(f"  正常多边形: {normal_anns}")
    print(f"  线段标注: {line_anns}")
    return line_anns

# ==================== 使用示例 ====================
if __name__ == "__main__":
    COCO_JSON_PATH = "coco_with_scaled/sample0_256/anno/scene_000000.json" 
    DENSITY_IMAGE_PATH = "coco_with_scaled/sample0_256/density_map/scene_000000.png"
    
    # 先检查标注情况
    print("=== 标注信息检查 ===")
    check_and_count_annotations(COCO_JSON_PATH)
    
    # 执行可视化
    print("\n=== 开始可视化 ===")
    visualize_coco_on_density_with_lines(
        coco_json_path=COCO_JSON_PATH,
        density_image_path=DENSITY_IMAGE_PATH,
        output_path="visualized_result.png",
        alpha=0.4,
        line_thickness=2
    )