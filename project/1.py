import os
import shutil

# 刪除檔案
files_to_delete = ['data.pickle', 'model.p']

for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f"{file} 已刪除")
    else:
        print(f"{file} 不存在")

# 刪除資料夾
folder_to_delete = 'data'

if os.path.exists(folder_to_delete):
    shutil.rmtree(folder_to_delete)
    print(f"{folder_to_delete} 資料夾已刪除")
else:
    print(f"{folder_to_delete} 資料夾不存在")
