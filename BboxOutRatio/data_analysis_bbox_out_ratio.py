import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 配置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ========== 1. 数据加载与清理 ==========
file_path = 'output/out_bbox_out_ratio/layout.csv'
df = pd.read_csv(file_path)

# 数据质量检查
print("=== 数据质量检查 ===")
total_rows = len(df)
unique_rooms_raw = df['source_file'].nunique()
nan_rooms = df['source_file'].isna().sum()
duplicate_rows = df.duplicated().sum()

print(f"总行数: {total_rows}")
print(f"唯一房间数 (含NaN): {unique_rooms_raw}")
print(f"source_file 空值数: {nan_rooms}")
print(f"重复行数: {duplicate_rows}")

# 清理：删除 source_file 为空的行
df_clean = df.dropna(subset=['source_file']).copy()
print(f"清理后行数: {len(df_clean)}")
print(f"清理后唯一房间数: {df_clean['source_file'].nunique()}\n")

# ========== 2. 基础统计：全内部房间比例 ==========
room_internal_status = df_clean.groupby('source_file')['practically_inside'].all()

num_rooms_all_inside = room_internal_status.sum()
num_total_rooms = len(room_internal_status)
ratio_all_inside = num_rooms_all_inside / num_total_rooms

print("=== 基础统计 ===")
print(f"Number of rooms with all bboxes inside: {num_rooms_all_inside:,}")
print(f"Number of total rooms: {num_total_rooms:,}")
print(f"Ratio of rooms with all bboxes inside: {ratio_all_inside:.2%}\n")

# ========== 3. 异常值检测与分析 ==========
# 计算每个房间的面积统计
room_area_stats = df_clean.groupby('source_file').agg(
    total_outside_area=('outside_area', 'sum'),
    total_bbox_area=('total_area', 'sum'),
    bbox_count=('source_file', 'size')  # 每个房间的bbox数量
).reset_index()

# 剔除无效房间
room_area_stats = room_area_stats[room_area_stats['total_bbox_area'] > 0]
room_area_stats['outside_to_total_ratio'] = (
    room_area_stats['total_outside_area'] / room_area_stats['total_bbox_area']
)

# 定义极端异常阈值
EXTREME_THRESHOLD = 0.9

# 筛选异常房间
extreme_rooms = room_area_stats[
    room_area_stats['outside_to_total_ratio'] > EXTREME_THRESHOLD
].copy()

print("=== 异常值分析 ===")
print(f"极端异常房间数 (ratio > {EXTREME_THRESHOLD}): {len(extreme_rooms)}")
print(f"占总数比例: {len(extreme_rooms) / len(room_area_stats):.2%}")

# 异常房间统计特征
if not extreme_rooms.empty:
    print(f"\n异常房间统计特征:")
    print(f"  - 平均 bbox 数量: {extreme_rooms['bbox_count'].mean():.1f}")
    print(f"  - 中位数值: {extreme_rooms['outside_to_total_ratio'].median():.2%}")
    print(f"  - 最大比值: {extreme_rooms['outside_to_total_ratio'].max():.2%}")
    
    # 显示最异常的10个房间
    print(f"\n最异常的10个房间：")
    print(extreme_rooms.sort_values('outside_to_total_ratio', ascending=False)[
        ['source_file', 'outside_to_total_ratio', 'bbox_count', 'total_outside_area']
    ].head(10).to_string(index=False))

# ========== 4. 详细调查特定异常房间 ==========
def investigate_room(room_name):
    """调查指定房间的bbox详情"""
    room_data = df_clean[df_clean['source_file'] == room_name]
    print(f"\n=== 房间 '{room_name}' 的详细分析 ===")
    print(f"bbox总数: {len(room_data)}")
    print(f"完全在内数量: {room_data['practically_inside'].sum()}")
    print(f"超出数量: {(~room_data['practically_inside']).sum()}")
    
    if not room_data.empty:
        print("\n前10个bbox详情:")
        display_cols = ['bbox_id', 'total_area', 'outside_area', 
                       'practically_inside', 'overlap_ratio'] if 'overlap_ratio' in room_data.columns else [
                       'total_area', 'outside_area', 'practically_inside']
        print(room_data[display_cols].head(10).to_string(index=False))

# 示例：调查最异常的房间（可选）
if not extreme_rooms.empty:
    most_extreme_room = extreme_rooms.iloc[0]['source_file']
    investigate_room(most_extreme_room)

# ========== 5. 可视化 ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 超出比例分布直方图
sns.histplot(room_area_stats['outside_to_total_ratio'], bins=200, kde=True, ax=axes[0,0], color='blue')
axes[0,0].set_title('Outside Area Ratio Distribution')
axes[0,0].set_xlabel('Outside / Total Ratio')
axes[0,0].set_ylabel('Room Count')
axes[0,0].axvline(EXTREME_THRESHOLD, color='red', linestyle='--', lw=2, label=f'Threshold {EXTREME_THRESHOLD}')
axes[0,0].legend()

# 图2: 累积分布函数 (CDF)
sorted_ratio = room_area_stats['outside_to_total_ratio'].sort_values()
y = np.arange(len(sorted_ratio)) / len(sorted_ratio)
axes[0,1].plot(sorted_ratio, y, color='purple')
axes[0,1].axvline(EXTREME_THRESHOLD, color='red', linestyle='--', lw=2)
axes[0,1].set_title('CDF: Cumulative Distribution')
axes[0,1].set_xlabel('Outside / Total Ratio')
axes[0,1].set_ylabel('Cumulative Proportion')
axes[0,1].grid(True, alpha=0.3)

# 图3: 异常房间的bbox数量分布
if not extreme_rooms.empty:
    sns.histplot(extreme_rooms['bbox_count'], bins=30, ax=axes[1,0], color='red')
    axes[1,0].set_title('Extreme Rooms: BBox Count Distribution')
    axes[1,0].set_xlabel('Number of BBoxes per Room')
    axes[1,0].set_ylabel('Extreme Room Count')
else:
    axes[1,0].text(0.5, 0.5, 'No extreme rooms found', ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Extreme Rooms: BBox Count Distribution')

# 图4: 异常比例 vs bbox数量散点图
axes[1,1].scatter(room_area_stats['bbox_count'], room_area_stats['outside_to_total_ratio'], 
                 alpha=0.5, s=10, color='blue')
axes[1,1].scatter(extreme_rooms['bbox_count'], extreme_rooms['outside_to_total_ratio'], 
                 alpha=0.8, s=20, color='red', label=f'Extreme (>{EXTREME_THRESHOLD})')
axes[1,1].axhline(EXTREME_THRESHOLD, color='red', linestyle='--', lw=1)
axes[1,1].set_title('Room BBox Count vs Outside Ratio')
axes[1,1].set_xlabel('Number of BBoxes')
axes[1,1].set_ylabel('Outside / Total Ratio')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# ========== 6. 导出异常房间列表 ==========
if not extreme_rooms.empty:
    output_path = 'extreme_rooms_analysis.csv'
    extreme_rooms.to_csv(output_path, index=False)
    print(f"\n异常房间列表已导出到: {output_path}")

# ========== 7. 总体统计摘要 ==========
print("\n=== 总体统计摘要 ===")
stats_df = room_area_stats['outside_to_total_ratio'].describe()
print(stats_df.to_string())