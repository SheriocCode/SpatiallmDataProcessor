import numpy as np
import cv2
import open3d as o3d
import glob
import os
import pandas as pd
from collections import defaultdict
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm

SAMPLE_ID = 0


def generate_2d_point_cloud_density_map(points, output_png_path, target_size=(256, 256)):
    if points.shape[1] == 3:
        points = points[:, :2]

    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    range_xy = max_xy - min_xy

    max_xy = max_xy + 0.1 * range_xy
    min_xy = min_xy - 0.1 * range_xy
    range_xy = max_xy - min_xy

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

# ========== 子进程：返回结果而非写日志 ==========
def process_single_scene(scene_data, data_root, output_dir, target_size):
    """处理单个场景，返回结果字典"""
    scene_id, chunk_ids = scene_data
    result = {
        'scene_id': scene_id,
        'success': False,
        'error': None,
        'chunks_processed': 0,
        'files_processed': 0,
        'total_points': 0,
        'warn_messages': []  # 收集警告信息
    }
    
    try:
        all_points = []
        
        for chunk_id in chunk_ids:
            chunk_folder = os.path.join(data_root, f"chunk_{str(chunk_id).zfill(3)}")
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
        output_png = os.path.join(output_dir, f"{scene_id}.png")
        generate_2d_point_cloud_density_map(all_points, output_png, target_size)
        
        result['success'] = True
        result['total_points'] = len(all_points)
        
    except Exception as e:
        result['error'] = f"未预期错误: {str(e)}"
    
    return result

# ========== 主进程：并行处理 + 统一日志 ==========
def batch_generate_scene_density_maps_parallel(
    split_csv_path="split.csv",
    data_root="data",
    output_dir="output_density_maps",
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
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 配置进程数
    if num_workers is None:
        # I/O密集型任务可设为CPU核心数1.5倍
        num_workers = max(1, int(cpu_count() * 1.5))
    logging.info(f"启动 {num_workers} 个进程并行处理")
    
    # 4. 创建进程池
    process_func = partial(
        process_single_scene,
        data_root=data_root,
        output_dir=output_dir,
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
        
        # 记录警告信息（如果有）
        for warn in result['warn_messages']:
            logging.warning(f"[{scene_id}] {warn}")
        
        if result['success']:
            success_count += 1
            logging.info(
                f"✅ 场景 {scene_id} 成功: "
                f"处理chunks={result['chunks_processed']}, "
                f"文件={result['files_processed']}, "
                f"总点数={result['total_points']:,}"
            )
        else:
            logging.error(f"❌ 场景 {scene_id} 失败: {result['error']}")
    
    logging.info(f"🎉 完成！成功处理 {success_count}/{total_scenes} 个场景")


def run_for_params(sample_id, img_size):
    global SAMPLE_ID
    data_root = "/mnt/data3/spatial_dataset/pcd"
    SAMPLE_ID = sample_id
    IMG_SIZE = img_size
    
    output_dir = f"density_maps/sample{SAMPLE_ID}_{IMG_SIZE}"
    log_path = f'log/sample{SAMPLE_ID}_{IMG_SIZE}.log'
    split_csv_path = f"split_sample_{SAMPLE_ID}.csv"
    
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    NUM_WORKERS = 12

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        filemode='w'
    )
    logging.info(f"Starting for SAMPLE_ID={SAMPLE_ID}, IMG_SIZE={IMG_SIZE}")

    batch_generate_scene_density_maps_parallel(
        split_csv_path=split_csv_path,
        data_root=data_root,
        output_dir=output_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        num_workers=NUM_WORKERS
    )
    logging.info(f"Finished for SAMPLE_ID={SAMPLE_ID}, IMG_SIZE={IMG_SIZE}")

if __name__ == "__main__":
    PARAM_COMBINATIONS = [
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