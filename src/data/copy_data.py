import os
import shutil
import numpy as np
from tqdm import tqdm
import json

def copy_file():
    # 源文件夹路径
    source_dir = '/data/renruilong/datasets/scannet/scannet_data'
    # 目标文件夹路径
    target_dir = './src/data/3D_Instruct/scannet_pcls'
    os.makedirs(target_dir, exist_ok=True)
    # # 遍历源文件夹中的文件
    # for file_name in os.listdir(source_dir):
    #     # 判断文件是否以"_vert.npy"结尾
    #     if file_name.endswith('_vert.npy'):
    #         # 拼接源文件路径和目标文件路径
    #         source_path = os.path.join(source_dir, file_name)
    #         target_path = os.path.join(target_dir, file_name)
    #         # 复制文件到目标文件夹中
    #         shutil.copyfile(source_path, target_path)

    # 遍历源文件夹中的文件
    for file_name in tqdm(os.listdir(source_dir)):
        # 判断文件是否以".npy"结尾
        if file_name.endswith('_vert.npy') and 'align' not in file_name:
            # 拼接源文件路径和目标文件路径
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            
            # 加载文件数据
            data = np.load(source_path)
            
            # 提取[50000, 6]部分的数据
            new_data = data[:, :6]
            
            # 保存为新文件
            np.save(target_path, new_data)

    print('Done!!')

def copy_json_data():
    src_path = "./src/data/3D_Instruct/meta_file/vg_scanrefer_train_lamm.json"
    des_path = "./src/data/3D_Instruct/meta_file/LAMM_3dinstruct_11k_add_det3d_w_gpt4.json"
    outpath = "./src/data/3D_Instruct/meta_file/LAMM_3dinstruct_48k_add_gpt4_add_scanrefer.json"

    # 读取a.json和b.json
    with open(src_path, 'r') as f:
        data_a = json.load(f)

    with open(des_path, 'r') as f:
        data_b = json.load(f)

    # 将a中的数据添加到b中
    for item in tqdm(data_a):
        data_b.append(item)

    # 保存b.json
    with open(outpath, 'w') as f:
        json.dump(data_b, f, indent=4)

    print('Done!')

if __name__=='__main__':
    # copy_json_data()
    copy_file()