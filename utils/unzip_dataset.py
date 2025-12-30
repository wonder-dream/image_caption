import os
import zipfile
import subprocess
import shutil
import sys

# 尝试导入 tqdm，如果不存在则定义一个简单的替代品
try:
    from tqdm import tqdm
except ImportError:
    print("提示: 安装 tqdm 库可以显示漂亮的进度条 (pip install tqdm)")
    def tqdm(iterable=None, total=None, **kwargs):
        if iterable:
            return iterable
        class SimpleBar:
            def update(self, n=1): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return SimpleBar()

def unzip_dataset(zip_path="data/images.zip", extract_dir="data"):
    """
    解压数据集文件到指定目录。
    优先使用系统 unzip 命令加速并配合 tqdm 显示进度。
    """
    if not os.path.exists(zip_path):
        print(f"错误: 未找到zip文件 '{zip_path}'")
        return

    # 确保解压目录存在
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"已创建目录: '{extract_dir}'")

    print(f"正在读取 zip 文件信息: '{zip_path}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件列表用于计算总数和后备解压
            file_list = zip_ref.namelist()
            total_files = len(file_list)
    except zipfile.BadZipFile:
        print(f"错误: '{zip_path}' 不是一个有效的zip文件。")
        return

    print(f"准备解压 {total_files} 个文件到 '{extract_dir}'...")

    # 方法1: 尝试使用系统 unzip 命令 (速度快，通过管道获取进度)
    if shutil.which('unzip'):
        print("检测到系统 'unzip' 命令，正在使用它进行加速解压...")
        try:
            # -o: 覆盖不提示
            # -DD: 不恢复任何时间戳
            # -d: 指定输出目录
            # 注意：不使用 -q，因为我们需要读取 stdout 来更新进度条
            process = subprocess.Popen(
                ['unzip', '-o', '-DD', zip_path, '-d', extract_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1 # 行缓冲
            )
            
            with tqdm(total=total_files, unit="file", desc="系统unzip解压") as pbar:
                # 实时读取 stdout 更新进度
                for _ in process.stdout:
                    pbar.update(1)
            
            # 等待进程结束并获取返回码
            return_code = process.wait()
            stderr_output = process.stderr.read()

            if return_code in [0, 1]:
                # 过滤警告
                if stderr_output:
                    ignored_warnings = ["cannot set modif./access times", "No such file or directory"]
                    filtered_errors = []
                    for line in stderr_output.splitlines():
                        if not any(w in line for w in ignored_warnings) and line.strip():
                            filtered_errors.append(line)
                    
                    if filtered_errors:
                        print("解压过程中的警告/错误:")
                        for line in filtered_errors:
                            print(line)
                            
                print("解压完成！")
                validate_extraction(extract_dir)
                return
            else:
                print(f"系统 unzip 命令返回错误代码 {return_code}")
                if stderr_output:
                    print(f"错误信息:\n{stderr_output}")
                print("尝试使用 Python zipfile...")
                
        except Exception as e:
            print(f"调用 unzip 出错 ({e})，尝试使用 Python zipfile...")
    else:
        print("未检测到系统 'unzip' 命令，将使用 Python 内置库。")

    # 方法2: 使用 Python zipfile 逐个解压 (兼容性好，有进度条)
    try:
        print("正在使用 Python zipfile 解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 使用 tqdm 显示进度
            for file in tqdm(file_list, desc="Python解压", unit="file"):
                zip_ref.extract(file, path=extract_dir)
                
        print("解压完成！")
        validate_extraction(extract_dir)

    except Exception as e:
        print(f"解压过程中发生错误: {e}")

def validate_extraction(extract_dir):
    """验证解压结果"""
    images_folder_path = os.path.join(extract_dir, 'images')
    if os.path.exists(images_folder_path) and os.path.isdir(images_folder_path):
        try:
            if any(os.scandir(images_folder_path)):
                 print(f"验证成功：'{images_folder_path}' 存在且不为空。")
            else:
                 print(f"警告: '{images_folder_path}' 存在但是空的。")
        except Exception:
             print(f"验证成功：'{images_folder_path}' 存在。")
    else:
        print(f"警告: 解压后未直接在 '{extract_dir}' 下找到 'images' 文件夹。请检查zip文件内容结构。")

if __name__ == "__main__":
    unzip_dataset()
