import os
import zipfile
from tqdm import tqdm

def unzip_dataset(zip_path="data/images.zip", extract_dir="data"):
    """
    解压数据集文件到指定目录。

    Args:
        zip_path (str): zip文件的路径。
        extract_dir (str): 解压的目标目录。
    """
    if not os.path.exists(zip_path):
        print(f"错误: 未找到zip文件 '{zip_path}'")
        return

    # 确保解压目录存在
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"已创建目录: '{extract_dir}'")

    print(f"正在解压 '{zip_path}' 到 '{extract_dir}'...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件列表以显示进度条
            file_list = zip_ref.infolist()
            with tqdm(total=len(file_list), unit='file', desc="解压进度") as pbar:
                for file in file_list:
                    zip_ref.extract(member=file, path=extract_dir)
                    pbar.update(1)
        print("\n解压完成！")
        
        # 验证解压后的images文件夹是否存在
        images_folder_path = os.path.join(extract_dir, 'images')
        if os.path.exists(images_folder_path) and os.path.isdir(images_folder_path):
            num_images = len(os.listdir(images_folder_path))
            print(f"在 '{images_folder_path}' 中找到 {num_images} 张图片。")
        else:
            print(f"警告: 解压后未直接在 '{extract_dir}' 下找到 'images' 文件夹。请检查zip文件内容结构。")

    except zipfile.BadZipFile:
        print(f"错误: '{zip_path}' 不是一个有效的zip文件。")
    except Exception as e:
        print(f"解压过程中发生错误: {e}")

if __name__ == "__main__":
    # 假设脚本在 image_caption 根目录下运行
    # zip文件路径为 'data/images.zip'
    # 解压到 'data/' 目录下, zip文件内的 'images/' 文件夹会被解压出来
    unzip_dataset()
