import os
import shutil
import re
import pandas as pd

import CSVProcessFns


df = pd.read_csv('data/split_raw.csv', sep=',')

# 按 scene_id 和 sample 统计唯一 room_id 的数量
grouped = df.groupby(['scene_id', 'sample'])
for value, group in grouped:
    print('value\n', value)
    print('group\n', group)
    break