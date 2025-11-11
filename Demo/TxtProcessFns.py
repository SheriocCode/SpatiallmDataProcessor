from typing import List, Union
import pathlib


def merge_ann_txt(txt_files: List[Union[str, pathlib.Path]],
                  out_file: Union[str, pathlib.Path],
                  encoding: str = 'utf-8') -> int:
    """
    将给定的多个标注 txt 文件合并为一个文件，自动跳过空行。

    参数
    ----
    out_file : str 或 pathlib.Path
        合并后的输出文件路径。
    txt_files : list[str | pathlib.Path]
        待合并的 txt 文件路径列表（非文件夹）。
    encoding : str, 可选
        文件读写编码，默认 'utf-8'。

    返回
    ----
    int
        实际写入的行数（不含空行）。
    """
    out_path = pathlib.Path(out_file)
    lines_written = 0

    with out_path.open('w', encoding=encoding) as fw:
        for f in map(pathlib.Path, txt_files):
            with f.open(encoding=encoding) as fr:
                for ln in fr:
                    ln = ln.rstrip()
                    if ln:               # 跳过空行
                        fw.write(ln + '\n')
                        lines_written += 1

    print(f'合并完成 -> {out_path}  共写入 {lines_written} 行')
    return lines_written