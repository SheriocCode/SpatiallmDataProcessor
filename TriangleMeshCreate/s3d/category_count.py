import json
from collections import Counter

# 1. 读取完整标注文件
with open('s3d.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 提取所有 annotations 中的 category_id
category_ids = [anno['category_id'] for anno in data['annotations']]
total_annos = len(category_ids)

# 3. 统计每个 category_id 出现的次数
category_counts = Counter(category_ids)

# 4. 建立 id -> name 映射
id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# 5. 输出结果（按 count 排序）
sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

print(f"Total annotations: {total_annos}\n")
print("S3d Category counts and percentages:")
for cat_id, count in sorted_counts:
    name = id_to_name.get(cat_id, f"Unknown({cat_id})")
    percentage = (count / total_annos) * 100
    print(f"{name}: {count} ({percentage:.2f}%)")