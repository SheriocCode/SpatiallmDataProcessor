import os
import glob
import re

data_dir = "data"
ply_dir = os.path.join(data_dir, "chunk000_pcd")
scene_id = "000000"
sample_id = "0"

# 文件名格式：scene_{scene_id}_{room_id}_{sample_id}.ply
pattern = f"scene_{scene_id}_*_{sample_id}.ply"
full_pattern = os.path.join(ply_dir, pattern)

target_plys = glob.glob(full_pattern)


# 从sample_{id}.csv中取出scene_id的全部room（可能分散在多个chunk中）
