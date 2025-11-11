from pathlib import Path
import re
from typing import List, Union

def merge_ply_files(ply_files: List[Union[str, Path]], output_ply: Union[str, Path]) -> None:
    """
    将给定路径列表中的二进制 PLY 文件合并成一个文件，要求所有文件 header 除顶点数外完全一致。

    参数
    ----
    ply_files : list[str | pathlib.Path]
        待合并的 .ply 文件路径列表（非文件夹）。
    output_ply : str 或 pathlib.Path
        合并后的输出 .ply 文件路径。

    返回
    ----
    None
    """
    def read_header(f):
        """读到 'end_header\\n' 为止，返回 (header_bytes, vertex_count)"""
        buf = b''
        vertex_count = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError('unexpected EOF in header')
            buf += line
            if line.strip() == b'end_header':
                break
            m = re.match(rb'element\s+vertex\s+(\d+)', line.strip())
            if m:
                vertex_count = int(m.group(1))
        return buf, vertex_count

    out_path = Path(output_ply)
    total_verts = 0
    header0 = None
    chunks = []

    for fname in map(Path, ply_files):
        with open(fname, 'rb') as f:
            hdr, n = read_header(f)
            if header0 is None:
                header0 = hdr
            else:
                a = re.sub(rb'element\s+vertex\s+\d+', b'element vertex 0', header0)
                b = re.sub(rb'element\s+vertex\s+\d+', b'element vertex 0', hdr)
                if a != b:
                    raise ValueError(f'header 不一致: {fname}')
            chunks.append(f.read())
            total_verts += n
            print(f'读取 {fname}  -> 顶点数 {n}')

    header_str = header0.decode('ascii')
    header_str = re.sub(r'(element\s+vertex\s+)\d+', r'\g<1>' + str(total_verts), header_str)
    header_bytes = header_str.encode('ascii')

    with open(out_path, 'wb') as fw:
        fw.write(header_bytes)
        for c in chunks:
            fw.write(c)

    print(f'合并完成 -> {out_path}  总文件数: {len(ply_files)}  总顶点数: {total_verts}')