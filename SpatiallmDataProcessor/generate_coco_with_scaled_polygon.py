import json
import numpy as np
import os
import logging
import tqdm
import time
import glob
import pandas as pd
import open3d as o3d
from shapely import Polygon
from collections import defaultdict
from functools import partial
import cv2
import math
from multiprocessing import Pool, cpu_count

from ply_2d_projection_with_txt_annotations import parse_annotation

def generate_2d_point_cloud_density_map(points, output_png_path, min_xy, range_xy, target_size=(256, 256)):
    """使用统一计算的min_xy和range_xy生成密度图"""
    if points.shape[1] == 3:
        points = points[:, :2]

    eps = 1e-6
    range_xy = np.where(range_xy < eps, eps, range_xy)

    normalized_xy = (points - min_xy) / range_xy
    pixel_coords = (normalized_xy * (np.array(target_size) - 1)).astype(np.int32)

    density = np.zeros(target_size, dtype=np.float32)
    h, w = target_size

    valid = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
    )
    valid_coords = pixel_coords[valid]
    unique_coords, counts = np.unique(valid_coords, axis=0, return_counts=True)

    density[unique_coords[:, 1], unique_coords[:, 0]] = counts
    density_normalized = density / (np.max(density) + 1e-6)
    density_uint8 = (density_normalized * 255).astype(np.uint8)
    cv2.imwrite(output_png_path, density_uint8)
    return density_uint8


def calculate_unified_scaling_params(point_coords, annotation_coords, target_size=(256, 256), padding=0.1):
    """
    合并点云和标注坐标计算统一的缩放参数
    point_coords: 点云坐标 (Nx2 numpy array)
    annotation_coords: 标注坐标 (Mx2 numpy array)
    返回: min_xy, range_xy, target_size
    """
    # 合并所有坐标
    all_coords = np.vstack([point_coords, annotation_coords])
    
    min_xy = np.min(all_coords, axis=0)
    max_xy = np.max(all_coords, axis=0)
    range_xy = max_xy - min_xy
    
    # 扩展边界
    max_xy = max_xy + padding * range_xy
    min_xy = min_xy - padding * range_xy
    range_xy = max_xy - min_xy
    
    eps = 1e-6
    range_xy = np.where(range_xy < eps, eps, range_xy)
    
    return min_xy, range_xy, target_size


def normalize_vector(x, y):
    length = math.hypot(x, y)
    if length == 0:
        return (0.0, 0.0)  # 避免除零，但正常情况下 wall 不会是零长
    return (x / length, y / length)

def process_single_scene(scene_data, split_csv_path, ply_data_root, layout_data_root, ply_output_dir, layout_output_dir, target_size):
    """处理单个场景，返回结果字典"""
    
    df = pd.read_csv(split_csv_path, sep=',', skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    scene_id, chunk_ids = scene_data
    result = {
        'scene_id': scene_id,
        'success': False,
        'error': None,
        'chunks_processed': 0,
        'files_processed': 0,
        'warn_messages': []
    }
    
    try:
        all_points = []
        all_annotation_coords = []  # 收集所有标注坐标
        
        # 1. 收集点云数据
        for chunk_id in chunk_ids:
            chunk_folder = os.path.join(ply_data_root, f"chunk_{str(chunk_id).zfill(3)}")
            if not os.path.exists(chunk_folder):
                result['warn_messages'].append(f"chunk文件夹不存在: {chunk_folder}")
                continue
            
            pattern = f"{scene_id}_*_{SAMPLE_ID}.ply"
            ply_files = glob.glob(os.path.join(chunk_folder, pattern))
            
            if not ply_files:
                result['warn_messages'].append(f"未找到PLY文件: {chunk_folder}/{pattern}")
                continue
            
            result['chunks_processed'] += 1
            
            for ply_path in ply_files:
                try:
                    pcd = o3d.io.read_point_cloud(ply_path)
                    if pcd.has_points():
                        points = np.asarray(pcd.points)
                        all_points.append(points[:, :2])
                        result['files_processed'] += 1
                    else:
                        result['warn_messages'].append(f"空点云: {ply_path}")
                except Exception as e:
                    result['warn_messages'].append(f"读取失败 {ply_path}: {e}")
                    continue
        
        if not all_points:
            result['error'] = "没有有效的点云数据"
            return result
        
        all_points = np.vstack(all_points)
        
        # 2. 收集标注数据坐标
        all_txt_files = []
        for chunk_id in chunk_ids:
            chunk_folder = os.path.join(layout_data_root, f"chunk_{str(chunk_id).zfill(3)}")
            if not os.path.exists(chunk_folder):
                result['warn_messages'].append(f"chunk文件夹不存在: {chunk_folder}")
                continue
            
            pattern = f"{scene_id}_*_{SAMPLE_ID}.txt"
            txt_files = glob.glob(os.path.join(chunk_folder, pattern))
            
            if not txt_files:
                result['warn_messages'].append(f"未找到txt文件: {chunk_folder}/{pattern}")
                continue
            
            all_txt_files.extend(txt_files)

        logging.info(f"[DEBUG] scene_id={scene_id}, layout_data_root={layout_data_root}")
        logging.info(f"[DEBUG] 所有找到的 txt_files: {all_txt_files}")
        
        # 临时存储标注坐标用于计算统一范围
        temp_annotation_coords = []
        temp_entities_list = []  # 存储所有实体用于后续处理
        
        for txt_path in all_txt_files:
            filename = os.path.basename(txt_path)
            if filename in TXT_REPAIRED:
                repaired_txt_path = os.path.join("poly_repair_output/txt_repaired/", filename)
                if os.path.exists(repaired_txt_path):
                    txt_path = repaired_txt_path
                    result['warn_messages'].append(f"✅ 使用修复后的文件 {filename}")
                    print(f'✅ 使用修复后的文件 {filename}')
                else:
                    result['warn_messages'].append(f"⚠️ 修复文件不存在")

            filename_without_ext = os.path.splitext(filename)[0]
            row = df.loc[df['id'] == filename_without_ext]
            if row.empty:
                result['warn_messages'].append(f"未找到对应的行: {txt_path}")
                continue
            
            room_id = int(row['room_id'].iloc[0])
            entities = parse_annotation(txt_path, room_id)
            temp_entities_list.append((entities, room_id, row))  # 保存实体和相关信息
            
            # 收集墙的坐标
            if entities["walls"]:
                for wall in entities["walls"]:
                    temp_annotation_coords.append([wall.ax, wall.ay])
                    temp_annotation_coords.append([wall.bx, wall.by])
            
            # 收集门的坐标
            if entities["doors"]:
                for door in entities["doors"]:
                    temp_annotation_coords.append([door.position_x, door.position_y])
            
            # 收集窗的坐标
            if entities["windows"]:
                for window in entities["windows"]:
                    temp_annotation_coords.append([window.position_x, window.position_y])
        
        # 3. 计算统一的缩放参数
        if not temp_annotation_coords:
            result['warn_messages'].append("未找到任何标注坐标，将仅使用点云范围")
            annotation_coords = np.array([[0, 0]])  # 用一个临时点避免空数组
        else:
            annotation_coords = np.array(temp_annotation_coords)
        
        # 计算统一缩放参数
        min_xy, range_xy, target_size = calculate_unified_scaling_params(
            all_points, 
            annotation_coords, 
            target_size
        )
        H, W = target_size
        
        # 4. 生成点云密度图（使用统一缩放参数）
        output_png = os.path.join(ply_output_dir, f"{scene_id}.png")
        generate_2d_point_cloud_density_map(all_points, output_png, min_xy, range_xy, target_size)
        
        # 定义缩放函数
        def scale_world_to_pixel(world_coords):
            world_coords = np.asarray(world_coords, dtype=np.float32)
            normalized = (world_coords - min_xy) / range_xy
            pixel_coords = (normalized * (np.array(target_size) - 1)).astype(np.int32)
            return pixel_coords
        
        # 5. 处理标注数据（使用统一缩放参数）
        annos = []
        for entities, room_id, row in temp_entities_list:
            room_type = row['room_type'].iloc[0]
            img_id = int(scene_id.split('_')[1])
            category_id = CATEGORIES_NAME_TO_ID[room_type]

            # 处理墙体
            if entities["walls"]:
                wall_vertices = []
                for wall in entities["walls"]:
                    wall_vertices.append([wall.ax, wall.ay])
                    wall_vertices.append([wall.bx, wall.by])
                
                scaled_wall_vertices = scale_world_to_pixel(wall_vertices)
                wall_polygon_points = scaled_wall_vertices[::2]
                wall_polygon = Polygon(wall_polygon_points)
                
                scaled_segmentation = []
                for point in wall_polygon_points:
                    x, y = point
                    scaled_segmentation.extend([float(x), float(y)])
                
                x_coords = scaled_wall_vertices[:, 0]
                y_coords = scaled_wall_vertices[:, 1]
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                expand = 2
                bbox = [
                    float(max(0, x_min - expand)),
                    float(max(0, y_min - expand)),
                    float(min(W, x_max - x_min + 2*expand)),
                    float(min(H, y_max - y_min + 2*expand))
                ]
                
                wall_anno = {
                    "room_id": room_id,
                    "segmentation": [scaled_segmentation],
                    "area": float(wall_polygon.area),
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": bbox,
                    "category_id": category_id,
                    "id": None
                }
                annos.append(wall_anno)
            
            # 处理门和窗
            def find_wall_by_id(walls, wall_id):
                for wall in walls:
                    if wall.id == wall_id:
                        return wall
                return None
            
            def process_line_entity(entity, entities, category_name):
                wall = find_wall_by_id(entities["walls"], entity.wall_id)
                if not wall:
                    logging.info(f"警告：{category_name} (id={entity.id}) 找不到对应的 wall (wall_id={entity.wall_id})")
                    return None
                
                cx, cy = entity.position_x, entity.position_y
                w = entity.width
                
                dir_x = wall.bx - wall.ax
                dir_y = wall.by - wall.ay
                dir_norm_x, dir_norm_y = normalize_vector(dir_x, dir_y)
                
                half_w = w / 2
                x1_world = cx - dir_norm_x * half_w
                y1_world = cy - dir_norm_y * half_w
                x2_world = cx + dir_norm_x * half_w
                y2_world = cy + dir_norm_y * half_w
                
                endpoints = np.array([[x1_world, y1_world], [x2_world, y2_world]])
                scaled_endpoints = scale_world_to_pixel(endpoints)
                x1, y1 = scaled_endpoints[0]
                x2, y2 = scaled_endpoints[1]
                
                segmentation = [float(x1), float(y1), float(x2), float(y2)]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                
                return {
                    "room_id": entity.room_id,
                    "segmentation": [segmentation],
                    "area": 0.0,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": bbox,
                    "category_id": CATEGORIES_NAME_TO_ID[category_name],
                    "id": None
                }
            
            if entities["doors"]:
                for door in entities["doors"]:
                    door_anno = process_line_entity(door, entities, "door")
                    if door_anno:
                        annos.append(door_anno)
            
            if entities["windows"]:
                for window in entities["windows"]:
                    window_anno = process_line_entity(window, entities, "window")
                    if window_anno:
                        annos.append(window_anno)

        # 添加annotation id
        for idx, anno in enumerate(annos):
            anno["id"] = idx
        
        scene_coco_data = {
            "images": [
                {
                    "file_name": f"{scene_id}.png",
                    "id": scene_id.split("_")[1],
                    "width": W,
                    "height": H
                }
            ],
            "annotations": annos,
            "categories": []
        }
        
        # 写入JSON
        try:
            logging.info(f'line323: scene_coco_data[{scene_id}]')
            logging.info(scene_coco_data)
            
            json_path = os.path.join(layout_output_dir, f"{scene_id}.json")
            logging.info(f"正在写入 JSON 文件: {json_path}")
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(scene_coco_data, f, ensure_ascii=False, indent=2)
            logging.info(f"✅ JSON 文件写入成功: {json_path}")
            
            result['success'] = True

        except Exception as e:
            result['error'] = f"写入 JSON 文件失败: {str(e)}"
            logging.error(f"[ERROR] 写入 JSON 失败: {e}")

    except Exception as e:
        result['error'] = f"未预期错误: {str(e)}"
    
    return result


def batch_generate_coco_scaled_parallel(    
    split_csv_path="split_sample_sample_id.csv",
    ply_data_root="data",
    layout_data_root="layout",
    ply_output_dir="ply_output_dir",
    layout_output_dir="layout_output_dir",
    target_size=(256, 256),
    num_workers=None
):
    # 1. 读取CSV
    if not os.path.exists(split_csv_path):
        logging.error(f"split.csv不存在: {split_csv_path}")
        return
    
    df = pd.read_csv(split_csv_path, sep=',', skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    
    scene_to_chunks = defaultdict(list)
    for _, row in df.iterrows():
        if pd.notna(row['scene_id']) and pd.notna(row['chunk_id']):
            scene_to_chunks[row['scene_id']].append(int(row['chunk_id']))
    
    for scene_id in scene_to_chunks:
        scene_to_chunks[scene_id] = list(set(scene_to_chunks[scene_id]))
    
    total_scenes = len(scene_to_chunks)
    logging.info(f"发现 {total_scenes} 个场景")
    
    # 2. 准备输出目录
    os.makedirs(ply_output_dir, exist_ok=True)
    os.makedirs(layout_output_dir, exist_ok=True)

    
    # 3. 配置进程数
    if num_workers is None:
        num_workers = max(1, int(cpu_count() * 1.5))
    logging.info(f"启动 {num_workers} 个进程并行处理")
    
    # 4. 创建进程池
    process_func = partial(
        process_single_scene,
        ply_data_root=ply_data_root,
        split_csv_path=split_csv_path,
        layout_data_root=layout_data_root,
        ply_output_dir=ply_output_dir,
        layout_output_dir=layout_output_dir,
        target_size=target_size
    )
    
    scene_items = list(scene_to_chunks.items())
    
    # 5. 并行处理
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(process_func, scene_items),
                total=len(scene_items),
                desc="处理进度"
            )
        )
    
    # 6. 统一记录日志
    success_count = 0
    for result in results:
        scene_id = result['scene_id']
        
        for warn in result['warn_messages']:
            logging.warning(f"[{scene_id}] {warn}")
        
        if result['success']:
            success_count += 1
            logging.info(
                f"✅ 场景 {scene_id} 成功: "
                f"处理chunks={result['chunks_processed']}, "
                f"文件={result['files_processed']}, "
            )
        else:
            logging.error(f"❌ 场景 {scene_id} 失败: {result['error']}")
    
    logging.info(f"🎉 完成！成功处理 {success_count}/{total_scenes} 个场景")


def run_for_params(sample_id, img_size):
    global SAMPLE_ID
    SAMPLE_ID = sample_id
    IMG_SIZE = img_size
    ply_data_root = "/mnt/data3/spatial_dataset/pcd" 
    layout_data_root = "/mnt/data3/spatial_dataset/layout"
    # ply_data_root = "data/pcd"
    # layout_data_root = "data/layout"

    ply_output_dir = f"coco_with_scaled/sample{SAMPLE_ID}_{IMG_SIZE}/density_map"
    layout_output_dir = f"coco_with_scaled/sample{SAMPLE_ID}_{IMG_SIZE}/anno"
    log_path = f'coco_with_scaled/log/sample{SAMPLE_ID}_{IMG_SIZE}.log'
    split_csv_path = f"data/csv/split_by_sample/split_sample_{SAMPLE_ID}.csv"

    os.makedirs(os.path.dirname(ply_output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(layout_output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    NUM_WORKERS = 12

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        filemode='w'
    )
    logging.info(f"Starting for SAMPLE_ID={SAMPLE_ID}, IMG_SIZE={IMG_SIZE}")

    batch_generate_coco_scaled_parallel(
        split_csv_path=split_csv_path,
        ply_data_root=ply_data_root,
        layout_data_root=layout_data_root,
        ply_output_dir=ply_output_dir,
        layout_output_dir=layout_output_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        num_workers=NUM_WORKERS
    )
    logging.info(f"Finished for SAMPLE_ID={SAMPLE_ID}, IMG_SIZE={IMG_SIZE}")

if __name__ == "__main__":
    # 读取categories.json
    file_path = 'categories.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        categories_json = json.load(f)
    categories = categories_json.get('categories', [])
    CATEGORIES = categories
    CATEGORIES_NAME_TO_ID = {}
    for category in categories:
        name = category.get('name')
        cat_id = category.get('id')
        if name is not None and cat_id is not None:
            CATEGORIES_NAME_TO_ID[name] = cat_id

    txt_poly_repaired_path = 'poly_repair_output/repaired_files_mapping.csv'
    
    def get_files_to_repair_list(repaired_mapping_csv_path):
        files_to_repair = []

        if not os.path.exists(repaired_mapping_csv_path):
            print(f"[警告] 修复文件映射 CSV 不存在: {repaired_mapping_csv_path}")
            return files_to_repair

        try:
            df = pd.read_csv(repaired_mapping_csv_path, sep=',', skipinitialspace=True)
            files_col = df.iloc[:, 1]

            for filename in files_col:
                if pd.notna(filename) and isinstance(filename, str):
                    files_to_repair.append(filename.strip())
        except Exception as e:
            print(f"[错误] 读取修复文件列表失败: {e}")

        return files_to_repair

    TXT_REPAIRED = get_files_to_repair_list(txt_poly_repaired_path)

    PARAM_COMBINATIONS = [
        {"sample_id": 0, "img_size": 256},
        {"sample_id": 0, "img_size": 1024},
        {"sample_id": 1, "img_size": 256},
        {"sample_id": 1, "img_size": 1024},
        {"sample_id": 2, "img_size": 256},
        {"sample_id": 2, "img_size": 1024},
        {"sample_id": 3, "img_size": 256},
        {"sample_id": 3, "img_size": 1024},
    ]

    for params in PARAM_COMBINATIONS:
        sample_id = params["sample_id"]
        img_size = params["img_size"]
        print(f"\n{'='*40}")
        print(f"Running for SAMPLE_ID={sample_id}, IMG_SIZE={img_size}")
        print(f"{'='*40}")
        run_for_params(sample_id, img_size)