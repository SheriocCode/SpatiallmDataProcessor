# 将接近垂直/水平的墙垂直、水平标准化
import os
import math
import json
import logging

logging.basicConfig(
    filename='.log',
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    filemode='w'
)

def check_and_fix_segmentation(segmentation, angle_tolerance=5):
    """
    检查线段角度并修复接近水平/垂直的极小偏差线段
    :param segmentation: 原始顶点列表，格式为[(x1,y1), (x2,y2), ...]
    :param angle_tolerance: 角度误差容忍度（度）
    :return: 修正后的顶点列表、修复记录
    """
    # 先检查无效线段
    invalid = check_segment_angles(segmentation, angle_tolerance)
    
    # 复制原始顶点用于修正（避免修改原列表）
    fixed_vertices = [list(vertex) for vertex in segmentation.copy()]  # 转为列表方便修改
    fix_records = []
    
    # 处理每个无效线段
    for seg in invalid:
        if seg['issue'] == "极小偏差（接近水平/垂直但不严格符合）":
            start_idx = seg['start_index']
            end_idx = seg['end_index']
            angle = seg['angle']
            (x1, y1) = seg['start_vertex']
            (x2, y2) = seg['end_vertex']
            
            # 判断接近水平（0°）还是垂直（90°）
            if abs(angle - 0) < abs(angle - 90):
                # 修复为水平线段（y坐标统一）
                # 取两点y的平均值（或可改为保留start/end的y，如y1）
                fixed_y = round((y1 + y2) / 2, 1)  # 保留1位小数避免浮点数误差
                # 更新两个顶点的y坐标
                fixed_vertices[start_idx][1] = fixed_y
                fixed_vertices[end_idx][1] = fixed_y
                fix_records.append(f"线段 {start_idx}→{end_idx}：修复为水平线（y={fixed_y}），原角度{angle}度")
            else:
                # 修复为垂直线段（x坐标统一）
                # 取两点x的平均值
                # fixed_x = round((x1 + x2) / 2, 1)  # 保留1位小数避免浮点数误差
                # 保留start的x1
                fixed_x = x1
                # 更新两个顶点的x坐标
                fixed_vertices[start_idx][0] = fixed_x
                fixed_vertices[end_idx][0] = fixed_x
                fix_records.append(f"线段 {start_idx}→{end_idx}：修复为垂直线（x={fixed_x}），原角度{angle}度")
    
    # 转换回元组列表
    fixed_vertices = [tuple(v) for v in fixed_vertices]
    return fixed_vertices, fix_records


# 复用之前的检查函数
def check_segment_angles(segmentation, angle_tolerance=5):
    vertices = segmentation.copy()
    vertices.append(vertices[0])
    invalid_segments = []
    
    for i in range(len(vertices) - 1):
        (x1, y1) = vertices[i]
        (x2, y2) = vertices[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        
        is_valid = False
        angle = None
        
        if dy == 0:
            is_valid = True
            angle = 0.0
        elif dx == 0:
            is_valid = True
            angle = 90.0
        else:
            rad = math.atan2(dy, dx)
            angle = math.degrees(rad) % 180
            if (abs(angle - 45) <= angle_tolerance) or (abs(angle - 135) <= angle_tolerance):
                is_valid = True
        
        if not is_valid:
            invalid_segments.append({
                'start_vertex': (x1, y1),
                'end_vertex': (x2, y2),
                'start_index': i,
                'end_index': (i + 1) % len(segmentation),
                'angle': round(angle, 4) if angle is not None else None,
                'issue': "极小偏差（接近水平/垂直但不严格符合）" if (
                    angle is not None and (abs(angle - 0) < angle_tolerance or abs(angle - 90) < angle_tolerance)
                ) else "角度不符合条件"
            })
    
    return invalid_segments


def process_single_file(json_path):
    # json_path = 'scene_000000.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data["annotations"]
    for anno in annotations:
        segmentation = anno["segmentation"][0]
        segmentation_id = anno["id"]
        # ==========================检查逻辑==========================
        # 转换为顶点列表
        test_segmentation = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]

        # 原始顶点列表
        raw_vertexs_list = []
        for idx, v in enumerate(test_segmentation):
            raw_vertexs_list.append((f'顶点{idx}', v))

        # 检查并修复
        fixed, records = check_and_fix_segmentation(test_segmentation)

        #修复后的顶点列表
        fixed_vertexs_list = []
        for idx, v in enumerate(fixed):
            fixed_vertexs_list.append((f'顶点{idx}', v))
        
        if not records:
            continue
        
        logging.info(f"====修复segmentation_id:{segmentation_id}====")
        logging.info('原始顶点列表:')
        logging.info(raw_vertexs_list)

        logging.warning("修复记录:")
        for record in records:
            logging.warning(record)
        
        logging.info('修复后的顶点列表:')
        logging.info(fixed_vertexs_list)

        # 验证修复结果（再次检查）
        invalid = check_segment_angles(fixed)
        if not invalid:
            continue
        
        logging.error('修复后依然存在问题')
        print("\n修复后仍存在问题线段：")
        for seg in invalid:
            logging.error(f"线段 {seg['start_index']}→{seg['end_index']}：角度{seg['angle']}度，问题：{seg['issue']}")
        

if __name__=='__main__':
    anno_dir = 'coco_with_scaled/sample0_256/anno'
    json_files_list = os.listdir(anno_dir)
    for json_file in json_files_list:
        json_path = os.path.join(anno_dir, json_file)
        logging.info(f'======Start Process File {json_file}=====')
        process_single_file(json_path)
