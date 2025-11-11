import numpy as np
import cv2
import open3d as o3d
import glob
import os
import re
import pandas as pd
from collections import defaultdict
import logging

# 配置日志：输出到文件，仅记录必要信息
logging.basicConfig(
    filename='.log',
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    filemode='w'
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


# ========== 您原来的函数，保持不变 ==========
def generate_2d_point_cloud_density_map(
    points,  # Nx2 or Nx3 numpy array, 只需要 x,y
    output_png_path,
    target_size=(256, 256)
):
    if points.shape[1] == 3:
        points = points[:, :2]  # 只取 x, y

    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    range_xy = max_xy - min_xy

    max_xy = max_xy + 0.1 * range_xy  # 最大值向外扩展10%
    min_xy = min_xy - 0.1 * range_xy  # 最小值向内收缩10%
    range_xy = max_xy - min_xy          # 重新计算扩展后的范围

    eps = 1e-6
    range_xy = np.where(range_xy < eps, eps, range_xy)  # 避免除以零

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

    density[unique_coords[:, 1], unique_coords[:, 0]] = counts  # y, x 顺序

    density_normalized = density / (np.max(density) + 1e-6)
    density_uint8 = (density_normalized * 255).astype(np.uint8)
    cv2.imwrite(output_png_path, density_uint8)
    print(f"✅ 合并房间密度图已保存: {output_png_path}，尺寸: {density_uint8.shape}")
    return density_uint8

# ========== 新增：批量处理逻辑 ==========

def batch_generate_scene_density_maps(
    split_csv_path="split.csv",
    data_root="data",
    output_dir="output_density_maps",
    target_size=(256, 256)
):
    # 1. 读取 split.csv
    if not os.path.exists(split_csv_path):
        print(f"❌ split.csv 文件不存在: {split_csv_path}")
        logging.error(f"split.csv 文件不存在: {split_csv_path}")
        return

    df = pd.read_csv(split_csv_path, sep=',', skipinitialspace=True)
    # 去除可能的空白列名和空白数据
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # 取出有用的列，确保字段存在
    required_columns = ['scene_id', 'chunk_id']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ split.csv 中缺少必要的列: {col}")
            logging.error(f"split.csv 中缺少必要的列: {col}")
            return

    # 按 scene_id 分组，得到每个 scene 对应的 chunk_id 列表
    scene_to_chunks = defaultdict(list)
    for _, row in df.iterrows():
        scene_id = row['scene_id']
        chunk_id = row['chunk_id']
        if pd.notna(scene_id) and pd.notna(chunk_id):
            scene_to_chunks[scene_id].append(chunk_id)

    # 去重
    for scene_id in scene_to_chunks:
        scene_to_chunks[scene_id] = list(set(scene_to_chunks[scene_id]))

    print(f"🔍 总共发现 {len(scene_to_chunks)} 个 unique scene_id")
    logging.info(f"总共发现 {len(scene_to_chunks)} 个 unique scene_id")

    # 2. 准备输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 遍历每个 scene_id
    for scene_id, chunk_ids in scene_to_chunks.items():
        all_points = []  # 收集该 scene 下所有 PLY 的 x,y 点

        print(f"\n🎯 正在处理 scene_id: {scene_id}，来自 chunk: {chunk_ids}")
        logging.info(f"正在处理 scene_id: {scene_id}，来自 chunk: {chunk_ids}")

        for chunk_id in chunk_ids:
            chunk_folder = os.path.join(data_root, f"chunk_{str(chunk_id).zfill(3)}")
            if not os.path.exists(chunk_folder):
                print(f"⚠️ chunk 文件夹不存在: {chunk_folder}")
                logging.warning(f"chunk 文件夹不存在: {chunk_folder}")
                continue

            # 匹配该 scene_id 下的所有 PLY 文件: scene_{scene_id}_*.ply
            pattern = f"{scene_id}_*_0.ply"
            ply_files = glob.glob(os.path.join(chunk_folder, pattern))

            if not ply_files:
                print(f"⚠️ 在 chunk {chunk_id} 中未找到 scene_id={scene_id} 的 PLY 文件")
                logging.warning(f"在 chunk {chunk_id} 中未找到 scene_id={scene_id} 的 PLY 文件")
                continue

            print(f"   📂 在 chunk {chunk_id} 中找到 {len(ply_files)} 个 PLY 文件")
            logging.info(f"在 chunk {chunk_id} 中找到 {len(ply_files)} 个 PLY 文件")

            for ply_path in ply_files:
                try:
                    pcd = o3d.io.read_point_cloud(ply_path)
                    if not pcd.has_points():
                        print(f"   ⚠️ 跳过空点云: {ply_path}")
                        logging.warning(f"跳过空点云: {ply_path}")
                        continue
                    points = np.asarray(pcd.points)  # Nx3
                    all_points.append(points[:, :2])  # 只要 x,y
                except Exception as e:
                    print(f"   ❌ 读取 PLY 文件失败 {ply_path}: {e}")
                    logging.error(f"读取 PLY 文件失败 {ply_path}: {e}")
                    continue

        if not all_points:
            print(f"❌ {scene_id} 没有有效的点云数据")
            logging.error(f"{scene_id} 没有有效的点云数据")
            continue

        all_points = np.vstack(all_points)  # 合并所有点

        # 4. 生成密度图
        output_png = os.path.join(output_dir, f"{scene_id}.png")
        generate_2d_point_cloud_density_map(all_points, output_png, target_size=target_size)
        print(f"   ✅ scene_id={scene_id} 的密度图已保存至: {output_png}")
        logging.info(f"scene_id={scene_id} 的密度图已保存至: {output_png}")

# ========== 执行入口 ==========
if __name__ == "__main__":
    split_csv_path = "split_sample_0.csv"         # 您的 scene-chunk 映射表
    data_root = "/mnt/data3/spatial_dataset/pcd"                   # 数据根目录，里面包含 chunk000, chunk001...
    output_dir = "output_density_maps"   # 输出密度图目录

    batch_generate_scene_density_maps(
        split_csv_path=split_csv_path,
        data_root=data_root,
        output_dir=output_dir,
        target_size=(256, 256)
    )