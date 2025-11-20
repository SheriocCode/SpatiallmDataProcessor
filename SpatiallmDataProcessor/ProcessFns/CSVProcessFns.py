import os
import csv
import pandas as pd

def extract_id_chunk_csv(root_dir: str, output_csv: str = 'id_chunk_csv.csv'):
    """
    遍历 data/layout 下的所有子文件夹，提取每个 chunk 文件夹中的所有文件的文件名 和所在的 chunk_id，
    并将结果写入 id_chunk_csv.csv 文件。

    root_dir : str
        存放 chunk_* 文件夹的根目录路径, e.g.(/mnt/data3/spatial_dataset/layout)
    [output_csv] : str, 可选
        输出 CSV 文件名（含路径），默认为 'id_chunk_csv.csv'。
        Format: 
            id(file_name) | chunk_id
            scene_000000_00_0 | 000
    return : None
    """
    rows = []

    for chunk_folder in os.listdir(root_dir):
        chunk_path = os.path.join(root_dir, chunk_folder)

        if os.path.isdir(chunk_path):
            # 提取 chunk_id，比如从 'chunk_000' 中得到 '000'
            chunk_id_num = chunk_folder.split('_')[1]  
            # 遍历该 chunk 文件夹中的所有文件
            for filename in os.listdir(chunk_path):
                if filename.endswith('.txt'):
                    # 获取不带后缀的文件名作为 id
                    file_id = os.path.splitext(filename)[0]
                    # 添加到结果列表
                    rows.append({
                        'id': file_id,
                        'chunk_id': chunk_id_num 
                    })

    # 写入 CSV 文件
    csv_columns = ['id', 'chunk_id']

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"已成功将所有 txt 文件的 id 和 chunk_id 写入 {output_csv}")


def add_chunk_id_to_split(split_csv: str = 'split.csv',
                          id_chunk_csv: str = 'id_chunk_csv.csv',
                          output_csv: str = 'split_raw.csv'):
    """
    将 id→chunk_id 映射表合并到原始 split.csv，生成带 chunk_id 的新文件。

    参数
    ----
    split_csv : str
        原始 split 数据文件路径（必须含 'id' 列）。
    id_chunk_csv : str
        包含 'id' 与 'chunk_id' 两列的映射表路径。
    output_csv : str
    返回
    ----
    None
    """
    # 读取两个 CSV 文件
    split_df = pd.read_csv(split_csv)             # 原始数据，要添加 chunk_id
    ids_chunk_df = pd.read_csv(id_chunk_csv, dtype={'chunk_id': str})  # 包含 id 和 chunk_id 的映射表

    ids_chunk_df['chunk_id'] = ids_chunk_df['chunk_id'].astype(str)

    # 使用 merge 进行左连接（left join），基于 id 列
    # 即：以 split_df 的 id 为准，去 ids_chunk_df 中找匹配的 chunk_id
    merged_df = pd.merge(
        split_df,
        ids_chunk_df,
        on='id',
        how='left'  # 左连接，保留 split.csv 所有行，匹配不到的 chunk_id 为 NaN
    )

    # [可选] 将chunk_id为空的行删除，并按照id进行排序得到split_raw.csv
    merged_df['chunk_id'] = merged_df['chunk_id'].fillna('')
    merged_df = merged_df[merged_df['chunk_id'] != '']
    merged_df = merged_df.sort_values(by=['id'])

    merged_df.to_csv(output_csv, index=False)
    print(f"已生成新的 CSV 文件：{output_csv}，已添加 chunk_id 列")


def scene_sample_room_counts(split_csv: str = 'split_raw.csv',
                               output_csv: str = 'scene_sample_room_counts.csv'):
    """
    读取原始 split 数据，统计每个 scene_id 下各 sample 的唯一 room_id 数量，
    并生成透视表：行=scene_id，列=sample{sample}_room_cnt，值=room 数量。
    """
    # 读取原始数据
    df = pd.read_csv(split_csv, sep=',')

    # 按 scene_id 和 sample 统计唯一 room_id 的数量
    room_counts = df.groupby(['scene_id', 'sample'])['room_id'].nunique().reset_index()
    room_counts.rename(columns={'room_id': 'room_cnt'}, inplace=True)

    # 构造透视表（pivot table）：行是 scene_id，列是 sample，值是 room_cnt
    pivot_df = room_counts.pivot(index='scene_id', columns='sample', values='room_cnt').fillna(0).astype(int)

    # 重命名列，使其符合输出字段要求
    pivot_df.columns = [f'sample{col}_room_cnt' for col in pivot_df.columns]

    # 重置索引以便导出为 DataFrame
    result_df = pivot_df.reset_index()

    # 保存结果
    result_df.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")

    # 展示前几行
    print(result_df.head())


def split_csv_group_by_sample(input_csv: str = 'data/split_raw.csv',
                               output_dir: str = 'data/split_by_sample'):
    """
    将 split.csv 按 sample 分组，并保存为单独的 CSV 文件。
    """
    # 读取原始 CSV 文件
    # input_file = 'data/split_raw.csv'

    input_file = input_csv
    df = pd.read_csv(input_file, sep=',')  

    # df['chunk_id'] = df['chunk_id'].astype(str).str.zfill(3) # 将 chunk_id 转换为字符串并填充前导 0 以确保长度为 3

    # 按 sample 分组
    grouped = df.groupby('sample')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个分组，保存为单独的 CSV 文件
    for sample_value, group in grouped:
        output_filename = os.path.join(output_dir, f'split_sample_{sample_value}.csv')
        group.to_csv(output_filename, index=False, sep=',') 
        print(f'已保存: {output_filename}')

    print("所有分组已保存完成。")

