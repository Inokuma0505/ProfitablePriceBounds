import os
import nbformat
from pathlib import Path

# スクリプト自身が置かれているディレクトリ
here = Path(__file__).parent

# ノートブックファイルへのパス（同じフォルダ内）
notebook_path = here / 'main_exp_get_all.ipynb'

# 出力先 src フォルダはプロジェクト直下に作りたいなら…
src_dir = here.parent / 'src'
src_dir.mkdir(exist_ok=True)

# .py 生成処理は同じ
script_path = src_dir / 'main_exp_get_all_two.py'
with open(script_path, 'w', encoding='utf-8') as f:
    f.write('# -*- coding: utf-8 -*-\n')
    f.write('"""\nAuto-generated script from main_exp_get_all.ipynb\n"""\n\n')
    nb = nbformat.read(str(notebook_path), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            f.write(cell.source + '\n\n')

print(f"Script has been created at: {script_path}")
